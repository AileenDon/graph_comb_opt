#ifndef MCF_ENV_H
#define MCF_ENV_H

#include "i_env.h"
#include <vector>
#include <set>
#include <map>

class McfEnv : public IEnv
{
public:
    McfEnv(double _norm); 

    virtual void s0(std::shared_ptr<Graph> _g) override; 

    virtual double step(int a) override; 

    virtual int randomAction() override; 

    virtual bool isTerminal() override; 

    virtual double getReward() override; 

    // MCF attributes
    std::vector<int> excess; // Nodes with excess flow that needs to be dispatched.
    std::vector<int> deficit; // Nodes with required flow intake.
    std::map<std::pair<int, int>, double> flows; // Flow amounts along the edges.
    std::map<std::pair<int, int>, double> costs; // Cost per unit flow on the edges.
};

#endif
