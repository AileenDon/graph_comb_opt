#include "config.h"

int cfg::max_bp_iter = 100;
int cfg::embed_dim = 64;
int cfg::dev_id = 0;
int cfg::batch_size = 32;
int cfg::max_iter = 1000;
int cfg::reg_hidden = 64;
int cfg::node_dim = 64;
int cfg::aux_dim = 10;
int cfg::min_n = 1;
int cfg::max_n = 100;
int cfg::mem_size = 1024;
int cfg::num_env = 10;
int cfg::n_step = 20;
Dtype cfg::learning_rate = 0.001;
Dtype cfg::l2_penalty = 0.0001;
Dtype cfg::momentum = 0.9;
Dtype cfg::w_scale = 0.1;
const char* cfg::save_dir = "./saved_models";

// MCF-specific parameters
Dtype cfg::default_capacity = 100.0;
Dtype cfg::default_cost = 1.0;
Dtype cfg::max_flow_cost = 1000000.0;
