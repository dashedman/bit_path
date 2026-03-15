use std::hash::{Hash, Hasher};
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyInt};

impl Hash for Bound<'_, PyAny> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        PyAnyMethods::hash(self).unwrap().hash(state);
    }
}


/// A Python module implemented in Rust.
#[pymodule]
pub mod rs_speedup {
    use std::collections::HashSet;
    use pyo3::prelude::*;
    use pyo3::types::{PyFrozenSet, PyInt, PySet};

    fn apply_conjunct(conjects: &mut HashSet<&Bound<'_, PyFrozenSet>>, item: &Bound<'_, PyFrozenSet>) {
        if conjects.contains(item) {
            conjects.remove(item);
        }
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn fast_xor_multiplication(conjects_a: &Bound<'_, PySet>, conjects_b: &Bound<'_, PySet>) -> PyResult<HashSet<>> {
        let mut merged_conjects = HashSet::new();

        for conj_a_any in conjects_a {
            let conj_a: &Bound<'_, PyFrozenSet> = conj_a_any.cast()?;
            for conj_b_any in conjects_b {
                let conj_b: &Bound<'_, PyFrozenSet> = conj_b_any.cast()?;

            }
        }

        // let mut intersection = HashSet::new();
        // let mut diff_a_b = HashSet::new();
        // let mut diff_b_a = HashSet::new();

        Ok(merged_conjects)
    }
}
