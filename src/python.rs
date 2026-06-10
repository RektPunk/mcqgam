use faer::{ColRef, MatRef};
use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub use crate::model::Regressor;
pub use crate::solver::admm::AdmmConfig;

/// NumPy view to Faer reference conversions.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
    type Faer = MatRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.nrows();
        let ncols = raw_arr.ncols();
        let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
        unsafe { MatRef::from_raw_parts(raw_arr.as_ptr(), nrows, ncols, strides[0], strides[1]) }
    }
}

impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
    type Faer = ColRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.len();
        let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
        unsafe { ColRef::from_raw_parts(raw_arr.as_ptr(), nrows, strides[0]) }
    }
}

/// Faer reference to NumPy view conversions.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

impl<'a, T> IntoNdarray for MatRef<'a, T> {
    type Ndarray = ArrayView2<'a, T>;

    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride() as usize;
        let col_stride = self.col_stride() as usize;
        unsafe {
            ArrayView2::from_shape_ptr(
                (nrows, ncols).strides((row_stride, col_stride)),
                self.as_ptr(),
            )
        }
    }
}

impl<'a, T> IntoNdarray for ColRef<'a, T> {
    type Ndarray = ArrayView1<'a, T>;

    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let row_stride = self.row_stride() as usize;
        unsafe { ArrayView1::from_shape_ptr(nrows.strides(row_stride), self.as_ptr()) }
    }
}

#[pyclass(name = "Regressor")]
pub struct PyRegressor {
    inner: Regressor,
}

#[pymethods]
impl PyRegressor {
    #[new]
    #[pyo3(signature = (degree=3, num_internal_knots=10, quantiles=vec![0.5]))]
    fn new(degree: usize, num_internal_knots: usize, quantiles: Vec<f64>) -> Self {
        let model = Regressor::new(degree, num_internal_knots, quantiles);
        Self { inner: model }
    }

    #[pyo3(signature = (x, y, max_iter=1000, max_inner_iter=10, rho=10.0, lambda=1e-4, epsilon=0.05, atol=1e-6, rtol=1e-6))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        max_iter: usize,
        max_inner_iter: usize,
        rho: f64,
        lambda: f64,
        epsilon: f64,
        atol: f64,
        rtol: f64,
    ) {
        let config = AdmmConfig::new(max_iter, max_inner_iter, rho, lambda, epsilon, atol, rtol);
        self.inner.fit(x.into_faer(), y.into_faer(), config);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        quantiles: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pred = self.inner.predict(x.into_faer(), quantiles);
        let py_pred = pred.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }
}
