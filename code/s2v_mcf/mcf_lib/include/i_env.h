#ifndef I_ENV_H
#define I_ENV_H

#include <vector>
#include <set>

#include "graph.h"

class IEnv
{
public:

    IEnv(double _norm) : norm(_norm), graph(nullptr) {}

    // Resets to initial state
    virtual void s0(std::shared_ptr<Graph> _g) = 0;

    // Runs a specific action
    // (steps into a specific numbered node)
    virtual double step(int a) = 0;

    // Runs a random action, changing the underlying state
    // Returns availability list value 
    // TODO: Understand the return value
    virtual int randomAction() = 0;

    // Checks if currently at the end state
    virtual bool isTerminal() = 0;

    // Gets the reward of the current state
    virtual double getReward() = 0;

    double norm;
    std::shared_ptr<Graph> graph;
    
    std::vector<std::vector<int>> state_seq;
    std::vector<int> act_seq, action_list;
    std::vector<double> reward_seq, sum_rewards;
};

#endif
