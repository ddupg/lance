// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[derive(Debug, Copy, Clone)]
pub(crate) enum PageType {
    Leaf,
    Branch,
}
