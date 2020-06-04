/*
<%
from sysconfig import get_path
cfg['include_dirs'] = [get_path('include') + '/pyTetris/']
cfg['compiler_args'] = ['-O3']
setup_pybind11(cfg)
%>
*/
#include<algorithm>
#include<vector>
#include<deque>
#include<unordered_map>
#include<iostream>
#include<pyTetris.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#define P 919

namespace py = pybind11;

const size_t n_actions = 7;
const size_t bStride = 200;

namespace std{
    template<>
    struct hash<Tetris>{
        size_t operator()(const Tetris &t) const{
            return t.hash();
        }
    };

    template<>
    struct hash<vector<char>>{
        size_t operator()(const vector<char> &v) const{
            size_t sum = 0;
            for(auto tmp : v)
                sum = P * sum + tmp;
            return sum;
        }
    };
}

class Agent{
    public:
        int current_episode;
        bool benchmark;
        Agent(){}
        int play(){return -1;}
        //int play(){std::cerr << "play not implemented" << std::endl; return -1;}:
        /*
        int get_action() {std::cerr << "get_action not implemented" << std::endl; return -1;};
        double* get_prob() {std::cerr << "get_prob not implemented" << std::endl; return 0;};
        void update_root(py::buffer game) {std::cerr << "update_root not implemented" << std::endl;};
        void close() {std::cerr << "close not implemented" << std::endl;};
        */
};

class TreeAgent: public Agent {
    public:
        int root;
        int sims;
        int max_nodes;
        py::object env;
        py::object env_args;
        int min_visits;
        bool projection;

        std::vector<int> child, visit, episode;
        std::vector<float> value, variance, score;
        std::vector<bool> end;

        Tetris game_tmp;
        std::vector<Tetris> games;

        std::vector<int> available, occupied;

        std::unordered_map<Tetris, int> node_index_map; 

        std::vector<int> visit_obs, node_to_obs;
        std::vector<float> value_obs, variance_obs;
        std::vector<bool> end_obs;
        std::vector<std::vector<char>> state_obs;

        std::vector<int> available_obs, occupied_obs;

        std::unordered_map<std::vector<char>, int> obs_index_map;

        std::vector<float> stats;

        py::dict arrays, obs_arrays;

        TreeAgent(int _sims, int _max_nodes, py::object _env, py::object _env_args, bool _projection, int _min_visits){
            sims = _sims;
            max_nodes = _max_nodes;
            env = _env;
            env_args = _env_args;
            projection = _projection;
            
            child.resize(max_nodes * n_actions);
            visit.resize(max_nodes);
            value.resize(max_nodes);
            variance.resize(max_nodes);
            episode.resize(max_nodes);
            score.resize(max_nodes);
            end.resize(max_nodes);
            games.resize(max_nodes);

            available.reserve(max_nodes);
            occupied.reserve(max_nodes);
            for(int i=1;i<max_nodes;++i)
                available.push_back(i);
            occupied.push_back(0);

            stats.resize(3 * n_actions);

            if(projection){
                node_to_obs.resize(max_nodes);
                state_obs.resize(max_nodes);
                visit_obs.resize(max_nodes);
                value_obs.resize(max_nodes);
                variance_obs.resize(max_nodes);
                end_obs.resize(max_nodes);
                
                available_obs.reserve(max_nodes);
                occupied_obs.reserve(max_nodes);
                for(int i=1;i<max_nodes;++i)
                    available_obs.push_back(i);
                occupied_obs.push_back(0);
            }
        }
        void compute_stats(int index){
            if(index == 0)
                index = root;
            
            for(int i=0;i<n_actions;++i){
                int c = child[root * n_actions + i];
                if(projection){
                    int o = node_to_obs[c]; 
                    stats[i] = visit_obs[o];
                    stats[i + n_actions] = value_obs[o] + score[c] - score[index];
                    stats[i + 2 * n_actions] = variance_obs[o];
                }else{
                    stats[i] = visit[c];
                    stats[i + n_actions] = value[c] + score[c] - score[index];
                    stats[i + 2 * n_actions] = variance[c];
                }
            }
        }

        int get_action(){
            compute_stats(root);
            auto max_it = std::max_element(&stats[n_actions], &stats[2 * n_actions]);
            return std::distance(&stats[n_actions], max_it);
        }

        std::vector<float> get_prob(){
            compute_stats(root);
            std::vector<float> result(n_actions);

            float sum = std::accumulate(stats.begin(), stats.begin() + n_actions, 0);
            
            for(int i=0;i<n_actions;++i)
                result[i] = stats[i] / sum;

            return result;
        }

        std::vector<float> get_value_and_variance(int node){
            if(node == 0)
                node = root;
            
            std::vector<float> result(2);
            if(projection){
                int o = node_to_obs[node];
                result[0] = value_obs[o];
                result[1] = variance_obs[o];
            }else{
                result[0] = value[node];
                result[1] = variance[node];
            }
        }

        void _expand(Tetris &g){
            int idx = _new_node(g);
            for(int i=0;i<n_actions;++i){
                game_tmp.copy_from(g);
                game_tmp.play(i);
                child[idx * n_actions + i] = _new_node(game_tmp);
            } 
        } 

        void expand(py::buffer &game){
            py::buffer_info info = game.request();

            Tetris *g = (Tetris*) (info.ptr);
            return _expand(*g);
           
        }

        int _new_node(Tetris &g){

            auto it = node_index_map.find(g);

            if(it == node_index_map.end()){
                
                if(available.empty())
                    remove_nodes();

                int idx = available.back();
                available.pop_back();

                games[idx].copy_from(g);

                episode[idx] = current_episode;
                score[idx] = g.score;

                node_index_map[games[idx]] = idx;

                occupied.push_back(idx);
                
                if(projection){
                    std::vector<char> state = g._getState();

                    auto o_it = obs_index_map.find(state);

                    if(o_it == obs_index_map.end()){
                        int o_idx = available_obs.back();
                        available_obs.pop_back();
                        state_obs[o_idx] = state;
                        end_obs[o_idx] = g.end;
                        obs_index_map[state] = o_idx;
                        occupied_obs.push_back(o_idx);
                        node_to_obs[idx] = o_idx;
                    }else{
                        node_to_obs[idx] = o_it->second;
                    }
                }
                
                return idx;
            }else{
                return it->second;
            }
        }

        int new_node(py::buffer &game){
            py::buffer_info info = game.request();

            Tetris *g = (Tetris*) (info.ptr);
            return _new_node(*g);
        }

        void update_root(py::buffer &game){

            py::buffer_info info = game.request();
            Tetris *g = (Tetris*) (info.ptr);

            root = _new_node(*g);

            if(g->end)
                current_episode += 1;
        }

        void update_available(){
            std::deque<size_t> to_traverse({root});

            std::unordered_set<size_t> traversed;

            for(size_t i=0; i < to_traverse.size(); ++i){
                auto pair = traversed.insert(to_traverse[i]);
                if(pair.second){
                    for(int c=0; c < n_actions; ++c){
                        size_t _c = child[*(pair.first) * n_actions + c];
                        if(traversed.find(_c) == traversed.end())
                            to_traverse.push_back(_c);
                    }
                }
            }

            occupied.resize(traversed.size());
            occupied.insert(occupied.begin(), traversed.begin(), traversed.end());

            available.resize(0);
            for(int i=0; i<max_nodes; ++i){
                if(std::find(occupied.begin(), occupied.end(), i) == occupied.end())
                    available.push_back(i);
            }

            std::cerr << "Number of occupied nodes: " << occupied.size() << std::endl;
            std::cerr << "Number of available nodes: " << available.size() << std::endl;

            if(projection){
                std::unordered_set<int> _o;
                for(auto v : occupied)
                    _o.insert(node_to_obs[v]);
                occupied_obs.resize(_o.size());
                occupied_obs.insert(occupied_obs.begin(), _o.begin(), _o.end());
                available_obs.resize(0);
                for(int i=0; i<max_nodes; ++i){
                    if(std::find(occupied_obs.begin(), occupied_obs.end(), i) == occupied_obs.end())
                        available_obs.push_back(i);
                }
                std::cerr << "Number of occupied observation nodes: " << occupied_obs.size() << std::endl;
                std::cerr << "Number of available observation nodes: " << available_obs.size() << std::endl;
            }
        }

        void reset_arrays(){

            for(auto v : available){
                node_index_map.erase(games[v]);
                visit[v] = 0;
                std::fill_n(&child[v * n_actions], n_actions, 0);
                episode[v] = 0;
                value[v] = 0;
                variance[v] = 0;
                score[v] = 0;
                end[v] = false;
            }
            if(projection){
                for(auto v : available_obs){
                    visit_obs[v] = 0;
                    value_obs[v] = 0;
                    variance_obs[v] = 0;
                    end_obs[v] = false;
                    obs_index_map.erase(state_obs[v]);
                    std::fill(state_obs[v].begin(), state_obs[v].end(), 0);
                }
            }
        }

        void remove_nodes(){


            std::cerr << "\nWARNING: REMOVING UNUSED NODES..." << std::endl;
            update_available();
        
            reset_arrays();         
        }
};

PYBIND11_MODULE(agent, m) {
    py::class_<Agent>(m, "Agent")
        .def(py::init<>());
    py::class_<TreeAgent, Agent>(m, "TreeAgent")
        //TreeAgent(int _sims, int _max_nodes, py::object _env, py::object _env_args, bool _projection, int _min_visits)
        .def(py::init<const int &, const int &, py::object &, py::object &, const bool &, const int &>())
        .def_readwrite("value", &TreeAgent::value)
        .def("remove_nodes", &TreeAgent::remove_nodes)
        .def("new_node", &TreeAgent::new_node)
        .def("update_available", &TreeAgent::update_available)
        .def("update_root", &TreeAgent::update_root)
        .def("compute_stats", &TreeAgent::compute_stats)
        .def("expand", &TreeAgent::expand);
}
