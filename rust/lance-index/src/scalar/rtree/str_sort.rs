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
use std::sync::Arc;

use crate::scalar::rtree::BBOX_SCHEMA;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::Result as DataFusionResult;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalSortExpr};
use futures::stream::unfold;
use futures::StreamExt;
use lance_datafusion::exec::OneShotExec;
use lance_datafusion::spill::create_replay_spill;
use tokio::sync::mpsc;

/// 使用 STR 算法对输入进行稳定重排的执行节点。
///
/// 行为：根据传入的 `center_x` 与 `center_y` 表达式计算点中心（或矩形中心），
/// 然后执行 STR 排序（先按 x 全局排序再切片，分片内按 y 排序），输出与输入同架构的记录集，
/// 仅改变行顺序，不增加/删除列。
#[derive(Debug)]
pub struct STRSortExec {
    input: Arc<dyn ExecutionPlan>,
    leaf_size: u32,
    spill_memory_limit: usize,
    properties: PlanProperties,
}

impl STRSortExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        leaf_size: u32,
        spill_memory_limit: usize,
    ) -> DataFusionResult<Self> {
        let schema = input.schema();
        let eq = EquivalenceProperties::new(schema.clone());
        let properties = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            leaf_size: leaf_size.max(1),
            spill_memory_limit,
            properties,
        })
    }

    fn tmp_path(&self, context: &TaskContext) -> DataFusionResult<RefCountedTempFile> {
        context
            .runtime_env()
            .disk_manager
            .create_tmp_file("strsort")
    }
}

impl DisplayAs for STRSortExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "STRSortExec: leaf_size={}, spill_memory_limit={}",
            self.leaf_size, self.spill_memory_limit
        )
    }
}

impl ExecutionPlan for STRSortExec {
    fn name(&self) -> &str {
        "STRSortExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> Arc<ArrowSchema> {
        BBOX_SCHEMA.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let child = children
            .into_iter()
            .next()
            .ok_or_else(|| DataFusionError::Plan("STRSortExec expects one child".into()))?;
        Ok(Arc::new(Self {
            input: child,
            leaf_size: self.leaf_size,
            spill_memory_limit: self.spill_memory_limit,
            properties: self.properties.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(
                "STRSortExec only supports a single partition".into(),
            ));
        }

        let schema = self.schema();
        let input = self.input.clone();
        let leaf_size = self.leaf_size;
        let spill_mem = self.spill_memory_limit;
        let spill_path_stage1 = self.tmp_path(&context)?;

        let (tx, rx) = mpsc::channel::<DataFusionResult<RecordBatch>>(1);

        let schema_clone = schema.clone();
        let spill_path_slice = self.tmp_path(&context)?;
        tokio::task::spawn({
            let context = context.clone();
            async move {
                // 一个辅助函数，便于统一错误处理：发送错误并返回
                let send_err = |tx: mpsc::Sender<DataFusionResult<RecordBatch>>,
                                err: DataFusionError| async move {
                    let _ = tx.send(Err(err)).await;
                };

                // 1) 预处理：计算中心并溢写为可回放的 spill（stage1）
                let input_stream = match input.execute(0, context.clone()) {
                    Ok(s) => s,
                    Err(e) => return send_err(tx, e).await,
                };
                let mut num_rows: usize = 0;

                // schema + [__str_cx, __str_cy]
                let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
                fields.push(Arc::new(ArrowField::new(
                    "__str_cx",
                    ArrowDataType::Float64,
                    false,
                )));
                fields.push(Arc::new(ArrowField::new(
                    "__str_cy",
                    ArrowDataType::Float64,
                    false,
                )));
                let augmented_schema = Arc::new(ArrowSchema::new(fields));

                let (mut spill_tx, spill_rx) = create_replay_spill(
                    spill_path_stage1.path().to_owned(),
                    augmented_schema.clone(),
                    spill_mem,
                );

                futures::pin_mut!(input_stream);
                while let Some(item) = input_stream.next().await {
                    let batch = match item {
                        Ok(b) => b,
                        Err(e) => return send_err(tx, e).await,
                    };
                    // BBOX_SCHEMA: [min_x, min_y, max_x, max_y, _rowid]
                    let column_mapping =
                        |index| batch.column(index).as_any().downcast_ref::<Float64Array>();
                    let min_x = column_mapping(0);
                    let min_y = column_mapping(1);
                    let max_x = column_mapping(2);
                    let max_y = column_mapping(3);
                    let (min_x, min_y, max_x, max_y) =
                        match (min_x, min_y, max_x, max_y) {
                            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                            _ => return send_err(
                                tx,
                                DataFusionError::Execution(
                                    "BBOX_SCHEMA must be Float64 columns (min_x,min_y,max_x,max_y)"
                                        .into(),
                                ),
                            )
                            .await,
                        };
                    let len = batch.num_rows();
                    let cx_arr = arrow_array::Float64Array::from_iter_values(
                        (0..len).map(|i| (min_x.value(i) + max_x.value(i)) * 0.5),
                    );
                    let cy_arr = arrow_array::Float64Array::from_iter_values(
                        (0..len).map(|i| (min_y.value(i) + max_y.value(i)) * 0.5),
                    );
                    let cx_f64: ArrayRef = Arc::new(cx_arr);
                    let cy_f64: ArrayRef = Arc::new(cy_arr);

                    let mut cols = batch.columns().to_vec();
                    cols.push(cx_f64);
                    cols.push(cy_f64);
                    let augmented = match RecordBatch::try_new(augmented_schema.clone(), cols) {
                        Ok(b) => b,
                        Err(e) => {
                            return send_err(tx, DataFusionError::ArrowError(e.into(), None)).await
                        }
                    };
                    num_rows += augmented.num_rows();
                    if let Err(e) = spill_tx.write(augmented).await {
                        return send_err(tx, e).await;
                    }
                }
                if let Err(e) = spill_tx.finish().await {
                    return send_err(tx, e).await;
                }

                // 2) 全局按 x 排序
                let sorted_by_x_stream: SendableRecordBatchStream = {
                    let source = Arc::new(OneShotExec::new(spill_rx.read()));
                    let key_index = augmented_schema.fields().len() - 2; // __str_cx
                    let sort_expr = PhysicalSortExpr {
                        expr: Arc::new(Column::new("__str_cx", key_index)),
                        options: arrow_schema::SortOptions::default(),
                    };
                    match SortExec::new([sort_expr].into(), source as Arc<dyn ExecutionPlan>)
                        .execute(0, context.clone())
                    {
                        Ok(s) => s,
                        Err(e) => return send_err(tx, e).await,
                    }
                };

                // 3) 计算切片参数
                let r = num_rows;
                let leaf = leaf_size as usize;
                let groups = (r + leaf - 1) / leaf; // ceil(r / n)
                let s_f = (groups as f64).sqrt().ceil() as usize;
                let stripes = s_f.max(1);
                let stripe_rows = (r + stripes - 1) / stripes; // ceil(r / stripes)

                // 4) 逐片处理并输出
                let mut carry_batches: Vec<RecordBatch> = Vec::new();
                let mut carry_rows: usize = 0;
                let mut sorted_x_stream = sorted_by_x_stream;

                loop {
                    while carry_rows < stripe_rows {
                        match sorted_x_stream.next().await {
                            Some(Ok(b)) => {
                                carry_rows += b.num_rows();
                                carry_batches.push(b);
                            }
                            Some(Err(e)) => return send_err(tx, e).await,
                            None => break,
                        }
                    }

                    if carry_batches.is_empty() {
                        break;
                    }

                    // 切片落盘
                    let (mut slice_tx, slice_rx) = create_replay_spill(
                        spill_path_slice.path().to_owned(),
                        augmented_schema.clone(),
                        spill_mem,
                    );
                    for b in carry_batches.drain(..) {
                        if let Err(e) = slice_tx.write(b).await {
                            return send_err(tx, e).await;
                        }
                    }
                    if let Err(e) = slice_tx.finish().await {
                        return send_err(tx, e).await;
                    }

                    // 切片内按 y 排序
                    let slice_sorted_stream: SendableRecordBatchStream = {
                        let slice_source = Arc::new(OneShotExec::new(slice_rx.read()));
                        let key_index_y = augmented_schema.fields().len() - 1; // __str_cy
                        let sort_expr_y = PhysicalSortExpr {
                            expr: Arc::new(Column::new("__str_cy", key_index_y)),
                            options: arrow_schema::SortOptions::default(),
                        };
                        match SortExec::new(
                            [sort_expr_y].into(),
                            slice_source as Arc<dyn ExecutionPlan>,
                        )
                        .execute(0, context.clone())
                        {
                            Ok(s) => s,
                            Err(e) => return send_err(tx, e).await,
                        }
                    };

                    futures::pin_mut!(slice_sorted_stream);
                    while let Some(item) = slice_sorted_stream.next().await {
                        let batch = match item {
                            Ok(b) => b,
                            Err(e) => return send_err(tx, e).await,
                        };
                        let keep_len = schema_clone.fields().len();
                        let cols = batch.columns()[..keep_len].to_vec();
                        let out = match RecordBatch::try_new(schema_clone.clone(), cols) {
                            Ok(b) => b,
                            Err(e) => {
                                return send_err(tx, DataFusionError::ArrowError(e.into(), None))
                                    .await
                            }
                        };
                        if tx.send(Ok(out)).await.is_err() {
                            return;
                        }
                    }

                    carry_rows = 0;
                }
            }
        });

        let out_stream = unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(item) => Some((item, rx)),
                None => None,
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            out_stream,
        )))
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics::default())
    }
}
