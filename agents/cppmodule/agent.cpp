/*
<%
from sysconfig import get_path
cfg['include_dirs'] = [get_path('include') + '/pyTetris/']
cfg['compiler_args'] = ['-O3', '-fvisibility=hidden']
setup_pybind11(cfg)
%>
*/
#include<algorithm>
#include<vector>
#include<deque>
#include<unordered_map>
#include<iostream>
#include<numeric>
#include<random>
#include<iterator>
#include<pyTetris.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<core.h>
#define P 919
#define SEED 123

namespace py = pybind11;

const size_t bStride = 200;

std::mt19937 mt(SEED);
std::uniform_real_distribution<double> unif(0., 1.);

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

class IntSampler{
    public:
        std::vector<int> indices;

        IntSampler(int n){
            indices.resize(n);
            std::iota(indices.begin(), indices.end(), 0);
        }

        std::vector<int> sample(int n){
            std::shuffle(indices.begin(), indices.end(), mt);
            return std::vector<int>(indices.begin(), indices.begin() + n);
        }
};

class Agent{
    public:
        int current_episode;
        bool benchmark;
        Agent(bool _benchmark){
            current_episode = 0;
            benchmark = _benchmark;
        }
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
        int root, max_nodes;
        bool projection;

        std::vector<std::array<int, n_actions>> child;
        std::vector<int> visit, episode;
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

        TreeAgent(int _max_nodes, bool _projection, bool _benchmark) : Agent(_benchmark) {
            max_nodes = _max_nodes;
            projection = _projection;
            
            child.resize(max_nodes);
            visit.resize(max_nodes, 0);
            value.resize(max_nodes, 0);
            variance.resize(max_nodes, 0);
            episode.resize(max_nodes, 0);
            score.resize(max_nodes, 0);
            end.resize(max_nodes, 0);
            games.resize(max_nodes);

            available.reserve(max_nodes);
            occupied.reserve(max_nodes);
            for(int i=1;i<max_nodes;++i)
                available.push_back(i);
            occupied.push_back(0);

            stats.resize(3 * n_actions, 0);

            if(projection){
                node_to_obs.resize(max_nodes, 0);
                state_obs.resize(max_nodes);
                visit_obs.resize(max_nodes, 0);
                value_obs.resize(max_nodes, 0);
                variance_obs.resize(max_nodes, 0);
                end_obs.resize(max_nodes, 0);
                
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
            
            for(size_t i=0;i<n_actions;++i){
                int c = child[root][i];
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
            
            for(size_t i=0;i<n_actions;++i)
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
            for(size_t i=0;i<n_actions;++i){
                game_tmp.copy_from(g);
                game_tmp.play(i);
                child[idx][i] = _new_node(game_tmp);
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

                if(available.empty())
                    std::cerr << "MAX_NODES EXCEEDED" << std::endl;

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
                    for(size_t c=0; c < n_actions; ++c){
                        size_t _c = child[*(pair.first)][c];
                        if(traversed.find(_c) == traversed.end())
                            to_traverse.push_back(_c);
                    }
                }
            }

            occupied.resize(traversed.size());
            occupied.insert(occupied.begin(), traversed.begin(), traversed.end());

            available.resize(0);
            for(int i=0; i<max_nodes; ++i){
                if(traversed.find(i) == traversed.end())
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
                    if(_o.find(i) == _o.end())
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
                std::fill_n(child[v].begin(), n_actions, 0);
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

        virtual void remove_nodes(){

            std::cerr << "\nWARNING: REMOVING UNUSED NODES..." << std::endl;
            update_available();
        
            reset_arrays();         
        }

        std::vector<char> fetch_observations(const std::vector<int> indices){
            std::vector<char> result(indices.size() * bStride);

            int offset = 0;
            for(auto idx : indices){
                std::copy(state_obs[idx].begin(), state_obs[idx].end(), result.begin() + offset);
                offset += bStride;
            }
            return result;
        }

        void get_unique_obs(size_t index, std::vector<int> &c, std::vector<int> &o){
            c.clear();
            o.clear();

            for(size_t i=0;i<n_actions;++i){
                int _c = child[index][i];
                if(_c == 0) continue;
                int _o = node_to_obs[_c];
                auto o_idx = std::find(o.begin(), o.end(), _o);
                if(o_idx == o.end()){
                    c.push_back(_c);
                    o.push_back(_o);
                }else{
                    int idx = o_idx - o.begin();
                    if(score[_c] > score[c[idx]])
                        c[idx] = _c;
                }
            }
        }
};

class MCTSAgent: public TreeAgent {
    public:
        int sims, max_nodes;
        bool leaf_parallelization;
        py::function evaluator;
        int evaluator_type;
        double gamma;
        MCTSAgent(int _sims, int _max_nodes, bool _projection, double _gamma, bool _benchmark, py::function _eval, int _eval_type, bool _LP): TreeAgent(_max_nodes, _projection, _benchmark){
            sims = _sims;
            leaf_parallelization = _LP;
            evaluator = _eval;
            evaluator_type = _eval_type;
            gamma = _gamma;
        }

        int play(){
            for(int i=0;i<sims;++i)
                mcts();
           
            compute_stats(root);
            return get_action();
        } 

        void mcts(){
            
            std::vector<int> trace = selection_obs(1);

            std::vector<int> c_nodes, c_obs;
            if(!games[trace.back()].end){
                _expand(games[trace.back()]);
                if(evaluator_type == 0){
                    if(leaf_parallelization){
                        get_unique_obs(trace.back(), c_nodes, c_obs);
                         
                        std::vector<char> f_obs = fetch_observations(c_obs);
                        
                        std::vector<size_t> shape = {c_nodes.size(), 1, 20, 10};

                        py::array_t<char> observations = py::array_t<char>(shape, &f_obs[0]);
                        
                        auto _result = evaluator(observations).cast<std::vector<py::object>>();
                        std::vector<float> __val = _result[0].cast<std::vector<float>>();
                        std::vector<float> __var = _result[1].cast<std::vector<float>>();
                        
                        backup_obs(trace, c_nodes, c_obs, __val, __var, false, true);
                    }else{
                    
                        std::vector<size_t> shape = {1, 1, 20, 10};
                        auto observation = games[trace.back()].getState();
                        observation.resize(shape);

                        auto result = evaluator(observation).cast<std::vector<float>>();

                        backup_obs_single(trace, score[trace.back()] + result[0], result[1]);
                    }


                }else if(evaluator_type == 1){
                    game_tmp.copy_from(games[trace.back()]);
                    py::object g_obj = py::cast(game_tmp);
                    std::vector<float> result = evaluator(g_obj).cast<std::vector<float>>();
                    float _val = result[0];
                    float _var = result[1];
                    backup_obs_single(trace, _val, _var); 
                }
            }else{
                backup_obs_single(trace, games[trace.back()].score, 0);
            }
        }

        std::vector<int> selection_obs(int low){
            std::vector<int> trace;
            int index = root;

            std::vector<float> _val, _var;
            std::vector<int> _vis, c_nodes, c_obs;
            while(true){
                trace.push_back(index);

                get_unique_obs(index, c_nodes, c_obs);
                
                if(c_nodes.empty()) break;

                int o = _check_low(c_obs, visit_obs, low);

                if(o == 0){
                    _vis.resize(c_nodes.size());
                    _val.resize(c_nodes.size());
                    _var.resize(c_nodes.size());
                    for(size_t i=0;i<c_nodes.size(); ++i){
                        int _o = c_obs[i], _c = c_nodes[i];
                        _vis[i] = visit_obs[_o];
                        _val[i] = value_obs[_o] + score[_c] - score[index];
                        _var[i] = variance_obs[_o];
                    }
                    index = policy_clt(c_nodes, _vis, _val, _var);
                }else{
                    auto o_it = std::find(c_obs.begin(), c_obs.end(), o);
                    index = c_nodes[o_it - c_obs.begin()];
                }
            }
            return trace;
        }

        void backup_obs_single(std::vector<int> &trace, float _val, float _var){
            for(int i=trace.size() - 1; i>=0; --i){
                int c_idx = trace[i];
                int o_idx = node_to_obs[c_idx];
                _val -= score[c_idx];
                if(visit_obs[o_idx] == 0){
                    value_obs[o_idx] = _val;
                    variance_obs[o_idx] = _var;    
                }else{
                    double delta = _val - value_obs[o_idx];
                    value_obs[o_idx] += delta / (visit_obs[o_idx] + 1);
                    double delta2 = _val - value_obs[o_idx];
                    variance_obs[o_idx] += (delta * delta2 - variance_obs[o_idx]) / (visit_obs[o_idx] + 1);
                }
                visit_obs[o_idx] += 1;
                _val = gamma * _val + score[c_idx];
            }
        }

        void backup_obs_single_mixture(std::vector<int> &trace, float _val, float _var){}
         
        void backup_obs(std::vector<int> &trace,
                    std::vector<int> &_child,
                    std::vector<int> &_obs,
                    std::vector<float> &_value,
                    std::vector<float> &_variance,
                    bool mixture,
                    bool averaged){

            if(_child.empty()){
                if(mixture)
                    backup_obs_single_mixture(trace, games[trace.back()].score, 0);
                else
                    backup_obs_single(trace, games[trace.back()].score, 0);
            }else{
                double _val_tmp, _var_tmp;
                _val_tmp = _var_tmp = 0;
                for(size_t i=0;i<_child.size(); ++i){
                    int __c = _child[i];
                    int __o = _obs[i];
                    if(visit_obs[__o] == 0){
                        visit_obs[__o] += 1;
                        if(end_obs[__o]){
                            value_obs[__o] = 0;
                            variance_obs[__o] = 0;
                        }else{
                            value_obs[__o] = _value[i];
                            variance_obs[__o] = _variance[i];
                        }
                    }
                    if(averaged){
                        _val_tmp += score[__c] + gamma * value_obs[__o];
                        _var_tmp += variance_obs[__o];
                    }else{
                        if(mixture)
                            backup_obs_single_mixture(trace, value_obs[__o] * gamma + score[__c], variance_obs[__o]);
                        else
                            backup_obs_single(trace, value_obs[__o] * gamma + score[__c], variance_obs[__o]);
                    }
                }
                if(averaged){
                    _val_tmp /= _child.size();
                    _var_tmp /= _child.size();
                    if(mixture)
                        backup_obs_single_mixture(trace, _val_tmp, _var_tmp);
                    else
                        backup_obs_single(trace, _val_tmp, _var_tmp);
                }
                  
            }
        }
        
};


class OnlineMCTSAgent: public MCTSAgent {
    public:
        bool online;

        int accumulation_policy;
        int memory_size, memory_growth_rate, memory_index;
        std::deque<int> nodes_per_episode;
        int accumulated_nodes, last_accumulation_episode;
        int episodes_per_train, last_training_episode;
        int n_trains;
        int min_visit;

        double memory_drop_prob;

        py::function train;
        py::array_t<float> m_state, m_value, m_variance, m_visit;

        OnlineMCTSAgent(int _sims, int _max_nodes, bool _online, int _accumulation_policy, int _memory_size, int _episodes_per_train, int _memory_growth_rate, int _min_visit, bool _projection, double _gamma, bool _benchmark, py::function _eval, int _eval_type, py::function _train, bool _LP): MCTSAgent(_sims, _max_nodes, _projection, _gamma, _benchmark, _eval, _eval_type, _LP){
            accumulation_policy = _accumulation_policy;

            memory_size = _memory_size;
            memory_index = 0;

            std::deque<int> nodes_per_episode;
            accumulated_nodes = 0;
            last_accumulation_episode = 0;
            memory_drop_prob = 0;

            min_visit = _min_visit;

            memory_growth_rate = _memory_growth_rate;
            n_trains = 0;

            episodes_per_train = _episodes_per_train;
            last_training_episode = 0;

            online = _online;

            if(online){
                m_state.resize({memory_size, 1, 20, 10});
                m_visit.resize({memory_size, 1});
                m_value.resize({memory_size, 1});
                m_variance.resize({memory_size, 1});
                train = _train;
            }

        }

        void remove_nodes(){

            std::cerr << "\nWARNING: REMOVING UNUSED NODES..." << std::endl;
            update_available();
        
            if(!benchmark && online){
               
                if(projection){
                    store_nodes(available_obs);
                }else{
                    store_nodes(available);
                }

                std::cerr << "Memory usage: " << memory_index << " / " << memory_size << std::endl;

                bool pass = false;
                if(accumulation_policy == 0){
                    if(last_accumulation_episode != current_episode){
                        nodes_per_episode.push_back(accumulated_nodes);
                        if(nodes_per_episode.size() > episodes_per_train)
                            nodes_per_episode.pop_front();
                        int sum = std::accumulate(nodes_per_episode.begin(), nodes_per_episode.end(), 0);
                        memory_drop_prob = std::max(0., 1. - double(memory_size) / sum);
                        
                        accumulated_nodes = 0;
                        last_accumulation_episode = current_episode;

                        std::cerr << "Average nodes stored per episode: " << sum / nodes_per_episode.size()
                            << "    Memory dropping probability: " << memory_drop_prob << std::endl;
                    }

                    int diff = current_episode - last_training_episode;
                    pass = diff >= episodes_per_train;
                    if(pass){
                        std::cerr << "Enough episodes (" << diff << " >= " << episodes_per_train << "), proceed to training." << std::endl;
                    }else{
                        if(memory_index >= memory_size){
                            std::cerr << "Memory limit exceeded, trimming memory." << std::endl;
                            random_trimming(0.01);
                            std::cerr << "Memory usage: " << memory_index << " / " << memory_size << std::endl;
                        }
                        std::cerr << "Not enough episodes (" << diff << " < " << episodes_per_train << "), collecting more episodes." << std::endl;
                    }
                }else if(accumulation_policy == 1){
                    int diff = current_episode - last_training_episode;
                    pass = diff >= episodes_per_train;
                    if(pass){
                        std::cerr << "Enough episodes (" << diff << " >= " << episodes_per_train << "), proceed to training." << std::endl;
                    }else{
                        if(memory_index >= memory_size){
                            double percentile = 0.01;
                            std::cerr << "Memory limit exceeded, trimming memory." << std::endl;
                            weighted_trimming(percentile);
                            std::cerr << "Memory usage: " << memory_index << " / " << memory_size << std::endl;
                        }
                        std::cerr << "Not enough episodes (" << diff << " < " << episodes_per_train << "), collecting more episodes." << std::endl;
                    }
                }else if(accumulation_policy == 2){
                    int diff = current_episode - last_training_episode;
                    pass = diff >= episodes_per_train || memory_index >= memory_size;
                    if(diff >= episodes_per_train){
                        std::cerr << "Enough episodes (" << diff << " >= " << episodes_per_train << "), proceed to training." << std::endl;
                    }else if(memory_index >= memory_size){
                        std::cerr << "Memory limit exceeded (" << memory_index << " >= " << memory_size << "), proceed to training." << std::endl;
                    }else{
                        std::cerr << "Not enough episodes (" << diff << " < " << episodes_per_train << "), collecting more episodes." << std::endl;
                    }
                }else if(accumulation_policy == 3){
                    int m_size = std::min(n_trains * memory_growth_rate, memory_size);
                    pass = memory_index >= m_size;
                    if(pass){
                        std::cerr << "Enough training data (" << memory_index << " >= " << m_size << "), proceed to training." << std::endl;
                    }else{
                        std::cerr << "Not enough training data (" << memory_index << " < " << m_size << "), collecting more data." << std::endl;
                    }
 
                }

                if(pass){
                    train(m_state, m_value, m_variance, m_visit, memory_index);
                    ++n_trains;
                    memory_index = 0;
                    last_training_episode = current_episode;
                    std::cerr << "Training complete." << std::endl;
                }

            }

            reset_arrays();         
        }

        void weighted_trimming(double percentile){
            std::vector<int> weights;
            weights.resize(memory_size);

            auto _vis = m_visit.mutable_unchecked<2>();
            auto _val = m_value.mutable_unchecked<2>();
            auto _var = m_variance.mutable_unchecked<2>();
            auto _state = m_state.mutable_unchecked<4>();

            for(int i=0;i<memory_size;++i)
                weights[i] = _vis(i, 0);
            
            std::sort(weights.begin(), weights.end());

            int threshold = weights[int(memory_size * percentile)];

            std::cerr << "Trimming memory with threshold " << threshold << "." << std::endl;

            int idx_fill = -1;
            for(int i=0;i<memory_size;++i){
                if(_vis(i, 0) <= threshold){
                    idx_fill = i;  
                    break;
                }
            }

            for(int i=idx_fill+1;i<memory_size;++i){
                if(_vis(i, 0) <= threshold){
                    --memory_index;
                    continue;
                }
                _vis(idx_fill, 0) = _vis(i, 0);
                _val(idx_fill, 0) = _val(i, 0);
                _var(idx_fill, 0) = _var(i, 0);
                for(int c=0;c<_state.shape(2);++c)
                    for(int r=0;r<_state.shape(3);++r)
                        _state(idx_fill, 0, c, r) = _state(i, 0, c, r);
                ++idx_fill;
            }
        }

        void random_trimming(double fraction){
            static IntSampler sampler(memory_size);
            std::vector<int> indices = sampler.sample(int(memory_size * fraction));
            std::sort(indices.begin(), indices.end());

            auto _vis = m_visit.mutable_unchecked<2>();
            auto _val = m_value.mutable_unchecked<2>();
            auto _var = m_variance.mutable_unchecked<2>();
            auto _state = m_state.mutable_unchecked<4>();

            int idx_fill = indices.front();
            for(auto it=indices.begin(); it!=indices.end(); ++it){
                int end = (it + 1) == indices.end()? memory_size : *(it+1);
                for(int start=*it+1; start < end; ++start){
                    _vis(idx_fill, 0) = _vis(start, 0);
                    _val(idx_fill, 0) = _val(start, 0);
                    _var(idx_fill, 0) = _var(start, 0);
                    for(int c=0;c<_state.shape(2);++c)
                        for(int r=0;r<_state.shape(3);++r)
                            _state(idx_fill, 0, c, r) = _state(start, 0, c, r);
                    ++idx_fill;
                }
            }
            memory_index -= indices.size();
        }

        void store_nodes(std::vector<int> &avail){
            std::cerr << "Storing unused nodes..." << std::endl;
        
            std::vector<float> *__val, *__var;
            std::vector<int> *__vis;
            std::vector<std::vector<char>> *__state;
            std::vector<bool> *__end;

            if(projection){
                __val = &value_obs;
                __var = &variance_obs;
                __vis = &visit_obs;
                __state = &state_obs;
                __end = &end_obs;
            }else{
                __val = &value;
                __var = &variance;
                __vis = &visit;
                __end = &end;
            }

            for(auto idx : avail){
                if((*__vis)[idx] < min_visit || (*__end)[idx]) continue;

                ++accumulated_nodes;
                if(accumulation_policy == 0 && unif(mt) < memory_drop_prob) continue;

                *m_visit.mutable_data(memory_index, 0) = (*__vis)[idx];
                *m_value.mutable_data(memory_index, 0) = (*__val)[idx];
                *m_variance.mutable_data(memory_index, 0) = (*__var)[idx];
                if(projection){
                    for(int c=0;c<m_state.shape(2);++c)
                        for(int r=0;r<m_state.shape(3);++r)
                            *m_state.mutable_data(memory_index, 0, c, r) = (*__state)[idx][m_state.shape(3) * c + r];
                }else{
                    for(int c=0;c<m_state.shape(2);++c)
                        for(int r=0;r<m_state.shape(3);++r)
                            *m_state.mutable_data(memory_index, 0, c, r) = *games[idx].getState().data(c, r);
                }
                if(++memory_index == memory_size) break;
            }

        }
};

PYBIND11_MODULE(agent, m) {
    py::class_<Agent>(m, "Agent")
        .def(py::init<const bool &>());
    py::class_<TreeAgent, Agent>(m, "TreeAgent")
        //TreeAgent(int _sims, int _max_nodes, py::object _env, py::object _env_args, bool _projection, int _min_visits)
        .def(py::init<const int &, const bool &, const int &>())
        .def("remove_nodes", &TreeAgent::remove_nodes)
        .def("new_node", &TreeAgent::new_node)
        .def("update_available", &TreeAgent::update_available)
        .def("update_root", &TreeAgent::update_root)
        .def("compute_stats", &TreeAgent::compute_stats)
        .def("expand", &TreeAgent::expand);
    py::class_<MCTSAgent, TreeAgent>(m, "MCTSAgent")
        .def(py::init<const int &, const int &, const bool &, const double &, const bool &, py::function &, const int &, const bool &>())
        .def("play", &MCTSAgent::play);
    py::class_<OnlineMCTSAgent, MCTSAgent>(m, "OnlineMCTSAgent")
        .def(py::init<int &, int &, bool &, int &, int &, int &, int &, int &, bool &, double &, bool &, py::function &, int &, py::function &, bool&>(),
             py::arg("sims") = 100, py::arg("max_nodes") = 100000, py::arg("online") = true, py::arg("accumulation_policy") = 0,
             py::arg("memory_size") = 500000, py::arg("episodes_per_train") = 25, py::arg("memory_growth_rate") = 5000,
             py::arg("min_visit") = 25, py::arg("projection") = true, py::arg("gamma") = 0.999, py::arg("benchmark") = false,
             py::arg("evaluator") = py::none(), py::arg("evaluation_type") = 0, py::arg("train") = py::none(), py::arg("LP") = true);
}
