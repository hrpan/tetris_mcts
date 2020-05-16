/*
<%
cfg['compiler_args'] = ['-O3']
setup_pybind11(cfg)
%>
*/
#include<unordered_set>
#include<deque>
#include<algorithm>
#include<cstdlib>
#include<special.h>
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

namespace py = pybind11;

const size_t n_actions = 7;

/*
    Utilities
*/
template<typename T>
void print_container(T &c){
    for(auto v : c){
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

std::unordered_set<size_t> get_all_childs(size_t index, py::array_t<int, 1> child){

    std::deque<size_t> to_traverse({index});

    std::unordered_set<size_t> traversed;

    for(size_t i=0; i < to_traverse.size(); ++i){
        auto pair = traversed.insert(to_traverse[i]);
        if(pair.second){
            for(size_t c=0; c < child.shape(1); ++c){
                size_t _c = *child.data(*pair.first, c);
                if(traversed.find(_c) == traversed.end())
                    to_traverse.push_back(_c);
            }
        }
    }

    return traversed;    
}

int check_low(const std::vector<int> &indices, const py::array_t<int, 1> &count, int n){
    auto count_uc = count.unchecked<1>();
    std::deque<int> low;
    for(int i : indices){
        if(count_uc(i) < n)
            low.push_back(i);
    }

    if(low.empty())
        return 0;
    else
        return low[rand() % low.size()];
}

/*
    POLICY
*/

int policy_clt(
        const std::vector<int> &nodes,
        const std::vector<int> &visit,
        const std::vector<float> &value,
        const std::vector<float> &variance){
    
    int n = std::accumulate(visit.begin(), visit.end(), 0);

    size_t max_idx = 0;
    float max_q = 0;
    float bound_coeff = norm_quantile(n);
    for(size_t i=0; i < value.size(); ++i){
        float q = value[i] + bound_coeff * sqrt(variance[i] / visit[i]);
        if(i == 0)
            max_q = q;
        else if(q > max_q){
            max_q = q;
            max_idx = i;
        }
    }

    return nodes[max_idx];
}

/*
    PROJECTION CORES
*/

void get_unique_child_obs(
        size_t index, 
        const py::array_t<int, 1> &child, 
        const py::array_t<float, 1> &score, 
        const py::array_t<int, 1> &n_to_o,
        std::vector<int> &c_nodes,
        std::vector<int> &c_obs){

    auto child_uc = child.unchecked<2>();
    auto score_uc = score.unchecked<1>();
    auto n_to_o_uc = n_to_o.unchecked<1>();

    c_nodes.clear();
    c_obs.clear();

    for(size_t i=0; i < n_actions; ++i){
        int c = child_uc(index, i);
        if(c == 0)
            continue;

        int o = n_to_o_uc(c);
        auto o_idx = std::find(c_obs.begin(), c_obs.end(), o);

        if(o_idx == c_obs.end()){
            c_nodes.push_back(c);
            c_obs.push_back(o);
        }else{
            size_t idx = std::distance(c_obs.begin(), o_idx);
            if(score_uc(c) > score_uc(c_nodes[idx]))
                c_nodes[idx] = c;
        }
    }

}

py::array_t<int, 1> select_trace_obs(
        size_t index, 
        py::array_t<int, 1> &child,
        py::array_t<int, 1> &visit,
        py::array_t<float, 1> &value,
        py::array_t<float, 1> &variance,
        py::array_t<float, 1> &score,
        py::array_t<int, 1> &n_to_o,
        size_t low){

    auto child_uc = child.unchecked<2>();
    auto visit_uc = visit.unchecked<1>();
    auto value_uc = value.unchecked<1>();
    auto variance_uc = variance.unchecked<1>();
    auto score_uc = score.unchecked<1>();
    auto n_to_o_uc = n_to_o.unchecked<1>();

    std::vector<int> trace;

    std::vector<int> _visit;
    std::vector<float> _value, _variance;

    _visit.reserve(n_actions);
    _value.reserve(n_actions);
    _variance.reserve(n_actions);

    std::vector<int> c_nodes, c_obs;
    c_nodes.reserve(n_actions);
    c_obs.reserve(n_actions);

    while(true){

        trace.push_back(index);

        get_unique_child_obs(index, child, score, n_to_o, c_nodes, c_obs);
        if(c_nodes.empty())
            break;
        auto o = check_low(c_obs, visit, low);

        size_t _s = c_nodes.size();
        if(o == 0){
            _visit.resize(_s);
            _value.resize(_s);
            _variance.resize(_s);
            for(size_t i=0; i < _s; ++i){
                int _o = c_obs[i];
                int _c = c_nodes[i];
                _visit[i] = visit_uc(_o);
                _value[i] = value_uc(_o) + score_uc(_c) - score_uc(index);
                _variance[i] = variance_uc(_o);
            }
            index = policy_clt(c_nodes, _visit, _value, _variance);
        }else{
            auto o_it = std::find(c_obs.begin(), c_obs.end(), o);
            index = c_nodes[std::distance(c_obs.begin(), o_it)];
        }
        
    }
    return py::array_t<int, 1>(trace.size(), &trace[0]);
}


/*
    PYBIND11
*/

PYBIND11_MODULE(core, m){
    m.def("get_all_childs", &get_all_childs);
    m.def("select_trace_obs", &select_trace_obs);
}
