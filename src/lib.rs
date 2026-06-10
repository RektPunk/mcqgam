use pyo3::prelude::*;

pub mod bspline;
pub mod interpolation;
pub mod model;
pub mod python;
pub mod solver;

#[pymodule]
fn mcqgam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyRegressor>()?;
    Ok(())
}
