// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! STRSortExec: 使用 STR (Sort-Tile-Recursive) 算法对地理数据进行外部排序。
//!
//! 参考：
//! - STR: A Simple and Efficient Algorithm for R-Tree Packing
//!   <https://www.cs.odu.edu/~mln/ltrs-pdfs/icase-1997-14.pdf>
//! - 中文介绍：STR树 —— R-tree的构建方案之一
//!   <https://www.cnblogs.com/fly2wind/p/14525405.html>

use std::any::Any;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::sorts::sort::SortExec as DFSortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, Statistics};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalSortExpr};
use datafusion_physical_expr::expressions::Column as DFColumnExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use futures::{StreamExt, TryStreamExt};
use lance_core::Result as LanceResult;

use crate::exec::OneShotExec;
use crate::spill::create_replay_spill;

/// 使用 STR 算法对输入进行稳定重排的执行节点。
///
/// 行为：根据传入的 `center_x` 与 `center_y` 表达式计算点中心（或矩形中心），
/// 然后执行 STR 排序（先按 x 全局排序再切片，分片内按 y 排序），输出与输入同架构的记录集，
/// 仅改变行顺序，不增加/删除列。
#[derive(Debug)]
pub struct STRSortExec {
    input: Arc<dyn ExecutionPlan>,
    center_x: Arc<dyn PhysicalExpr>,
    center_y: Arc<dyn PhysicalExpr>,
    leaf_size: usize,
    spill_memory_limit: usize,
    temp_dir: Option<PathBuf>,
    properties: PlanProperties,
}

impl STRSortExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        center_x: Arc<dyn PhysicalExpr>,
        center_y: Arc<dyn PhysicalExpr>,
        leaf_size: usize,
        spill_memory_limit: usize,
        temp_dir: Option<PathBuf>,
    ) -> DFResult<Self> {
        let schema = input.schema();
        let eq = EquivalenceProperties::new(schema.clone());
        let output_partitioning = Partitioning::UnknownPartitioning(1);
        let properties = PlanProperties::new(
            eq,
            output_partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            center_x,
            center_y,
            leaf_size: leaf_size.max(1),
            spill_memory_limit,
            temp_dir,
            properties,
        })
    }

    fn tmp_path(&self, context: &TaskContext, suffix: &str) -> PathBuf {
        if let Some(dir) = &self.temp_dir {
            dir.join(format!("{}_{}.arrows", self.plan_type(), suffix))
        } else {
            let tmp_dir = context.runtime_env().disk_manager().create_tmp_dir("strsort");
            tmp_dir.path().join(format!("{}_{}.arrows", self.plan_type(), suffix))
        }
    }

    fn plan_type(&self) -> &'static str { "str_sort" }
}

impl DisplayAs for STRSortExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "STRSortExec: leaf_size={}, spill_memory_limit={}",
                    self.leaf_size, self.spill_memory_limit
                )
            }
        }
    }
}

impl ExecutionPlan for STRSortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "STRSortExec"
    }

    fn schema(&self) -> Arc<ArrowSchema> {
        self.properties().schema().clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> DFResult<Arc<dyn ExecutionPlan>> {
        let child = children.into_iter().next().ok_or_else(|| DataFusionError::Plan("STRSortExec expects one child".into()))?;
        Ok(Arc::new(Self {
            input: child,
            center_x: self.center_x.clone(),
            center_y: self.center_y.clone(),
            leaf_size: self.leaf_size,
            spill_memory_limit: self.spill_memory_limit,
            temp_dir: self.temp_dir.clone(),
            properties: self.properties.clone(),
        }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution("STRSortExec only supports a single partition".into()));
        }

        let schema = self.schema();
        let input = self.input.clone();
        let cx = self.center_x.clone();
        let cy = self.center_y.clone();
        let leaf_size = self.leaf_size;
        let spill_mem = self.spill_memory_limit;
        let spill_path_stage1 = self.tmp_path(&context, "stage1");

        let stream = async_stream::try_stream! {
            // 1) 预处理：计算中心并溢写为可回放的 spill（stage1）
            let input_stream = input.execute(0, context.clone())?;
            let mut num_rows: usize = 0;

            // schema + [__str_cx, __str_cy]
            let mut fields = schema.fields().deref().clone();
            fields.push(Arc::new(ArrowField::new("__str_cx", ArrowDataType::Float64, false)));
            fields.push(Arc::new(ArrowField::new("__str_cy", ArrowDataType::Float64, false)));
            let augmented_schema = Arc::new(ArrowSchema::new(fields));

            let (mut spill_tx, spill_rx) = create_replay_spill(spill_path_stage1.clone(), augmented_schema.clone(), spill_mem);

            futures::pin_mut!(input_stream);
            while let Some(batch) = input_stream.next().await.transpose()? {
                let cx_arr = cx.evaluate(&batch)?.into_array(batch.num_rows())?;
                let cy_arr = cy.evaluate(&batch)?.into_array(batch.num_rows())?;
                let cx_f64: ArrayRef = if cx_arr.data_type() == &ArrowDataType::Float64 {
                    cx_arr
                } else {
                    Arc::new(cast(cx_arr.as_ref(), &ArrowDataType::Float64)?)
                };
                let cy_f64: ArrayRef = if cy_arr.data_type() == &ArrowDataType::Float64 {
                    cy_arr
                } else {
                    Arc::new(cast(cy_arr.as_ref(), &ArrowDataType::Float64)?)
                };

                let mut cols = batch.columns().to_vec();
                cols.push(cx_f64);
                cols.push(cy_f64);
                let augmented = RecordBatch::try_new(augmented_schema.clone(), cols)?;
                num_rows += augmented.num_rows();
                spill_tx.write(augmented).await?;
            }
            spill_tx.finish().await?;

            // 2) 全局按 x 排序
            let sorted_by_x_stream: SendableRecordBatchStream = {
                let source = Arc::new(OneShotExec::new(spill_rx.read()));
                // 排序键：__str_cx 列（最后倒数第二列）
                let key_index = augmented_schema.fields().len() - 2; // __str_cx
                let sort_expr = PhysicalSortExpr { expr: Arc::new(DFColumnExpr::new("__str_cx", key_index)), options: arrow_schema::SortOptions::default() };
                let sort_plan = DFSortExec::new(vec![sort_expr], source as Arc<dyn ExecutionPlan>);
                sort_plan.execute(0, context.clone())?
            };

            // 3) 根据 r 与 leaf_size 计算切片数 S 和每片大小
            let r = num_rows;
            let groups = (r + leaf_size - 1) / leaf_size; // ceil(r / n)
            let s_f = (groups as f64).sqrt().ceil() as usize;
            let stripes = s_f.max(1);
            let stripe_rows = (r + stripes - 1) / stripes; // ceil(r / stripes)

            // 4) 逐条带宽流式切片；每个切片内部再排序 y（外部可回放 + 排序）
            let mut carry_batches: Vec<RecordBatch> = Vec::new();
            let mut carry_rows: usize = 0;
            let mut sorted_x_stream = sorted_by_x_stream;

            loop {
                while carry_rows < stripe_rows {
                    match sorted_x_stream.next().await {
                        Some(Ok(b)) => { carry_rows += b.num_rows(); carry_batches.push(b); },
                        Some(Err(e)) => Err(e)?,
                        None => break,
                    }
                }

                if carry_batches.is_empty() {
                    break;
                }

                // 将切片写入临时 spill
                let spill_path_slice = self.tmp_path(&context, "slice");
                let (mut slice_tx, slice_rx) = create_replay_spill(spill_path_slice.clone(), augmented_schema.clone(), spill_mem);
                for b in carry_batches.drain(..) { slice_tx.write(b).await?; }
                slice_tx.finish().await?;

                // 对切片按 y 排序
                let slice_sorted_stream: SendableRecordBatchStream = {
                    let slice_source = Arc::new(OneShotExec::new(slice_rx.read()));
                    let key_index_y = augmented_schema.fields().len() - 1; // __str_cy
                    let sort_expr_y = PhysicalSortExpr { expr: Arc::new(DFColumnExpr::new("__str_cy", key_index_y)), options: arrow_schema::SortOptions::default() };
                    let sort_plan_y = DFSortExec::new(vec![sort_expr_y], slice_source as Arc<dyn ExecutionPlan>);
                    sort_plan_y.execute(0, context.clone())?
                };

                futures::pin_mut!(slice_sorted_stream);
                while let Some(batch) = slice_sorted_stream.next().await.transpose()? {
                    // 去掉辅助列，仅输出原始 schema
                    let keep_len = schema.fields().len();
                    let cols = batch.columns()[..keep_len].to_vec();
                    let out = RecordBatch::try_new(schema.clone(), cols)?;
                    yield out;
                }

                carry_rows = 0;
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema(), stream)))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(Statistics::default())
    }
}


