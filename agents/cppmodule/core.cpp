/*
<%
from sysconfig import get_path
cfg['include_dirs'] = [get_path('include') + '/pyTetris/']
cfg['compiler_args'] = ['-O3']
setup_pybind11(cfg)
%>
*/
#include<core.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

namespace py = pybind11;

/*
    PYBIND11
*/

PYBIND11_MODULE(core, m){
    m.def("get_all_childs", &get_all_childs);
    m.def("get_unique_child_obs", &get_unique_child_obs_);
    m.def("select_trace_obs", &select_trace_obs);
    m.def("backup_trace_obs", &backup_trace_obs);
    m.def("backup_trace_obs_LP", &backup_trace_obs_LP);
}
