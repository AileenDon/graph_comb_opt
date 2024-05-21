#ifndef Q_NET_H
#define Q_NET_H

#include "inet.h"
using namespace gnn;

class QNet : public INet
{
public:
    QNet();

    virtual void BuildNet() override;
    
    // Adapting the training setup to handle flows and costs
    virtual void SetupTrain(std::vector<int>& idxes, 
                            std::vector< std::shared_ptr<Graph> >& g_list, 
                            std::vector< std::vector<double>* >& flows,  // Changed from covered to flows
                            std::vector<int>& actions, 
                            std::vector<double>& target) override;

    // Setup for predictions across all nodes to decide the next flow adjustments
    virtual void SetupPredAll(std::vector<int>& idxes, 
                              std::vector< std::shared_ptr<Graph> >& g_list, 
                              std::vector< std::vector<double>* >& flows) override;  // Changed from covered to flows

    // Helper method to prepare graph input; might include flow conditions
    void SetupGraphInput(std::vector<int>& idxes, 
                         std::vector< std::shared_ptr<Graph> >& g_list, 
                         std::vector< std::vector<double>* >& flows,  // Changed from covered to flows
                         const int* actions);

    // Gets status info about graph related to flows and costs
    int GetStatusInfo(std::shared_ptr<Graph> g, int num, const double* flows, int& counter, std::vector<int>& idx_map); 

    // Sparse and dense tensors for selecting actions and representing global state, adapted for MCF
    SpTensor<CPU, Dtype> act_select, rep_global;
    SpTensor<mode, Dtype> m_act_select, m_rep_global;
    DTensor<CPU, Dtype> aux_feat;  
    DTensor<mode, Dtype> m_aux_feat;
    std::vector<int> avail_act_cnt;  /
};

#endif
