#include "config.h"
#include "mcf_lib.h"
#include "graph.h"
#include "nn_api.h"
#include "qnet.h"
#include "nstep_replay_mem.h"
#include "simulator.h"
#include "mcf_env.h"  
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

using namespace gnn;

void intHandler(int dummy) {
    exit(0);
}

int LoadModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");    
    net->model.Load(filename);
    return 0;
}

int SaveModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");
    net->model.Save(filename);
    return 0;
}

std::vector< std::vector<double>* > list_pred;
McfEnv* test_env;  
int Init(const int argc, const char** argv)
{
    signal(SIGINT, intHandler);
    
    cfg::LoadParams(argc, argv);
    GpuHandle::Init(cfg::dev_id, 1);

    net = new QNet();    
    net->BuildNet();

    NStepReplayMem::Init(cfg::mem_size);
    
    Simulator::Init(cfg::num_env);
    for (int i = 0; i < cfg::num_env; ++i)
        Simulator::env_list[i] = new McfEnv(cfg::max_n);  
    test_env = new McfEnv(cfg::max_n);  

    list_pred.resize(cfg::batch_size);
    for (int i = 0; i < cfg::batch_size; ++i)
        list_pred[i] = new std::vector<double>(cfg::max_n + 10);
    return 0;
}

// Further adaptations in functions interacting with graph nodes and training procedures
