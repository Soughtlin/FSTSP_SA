//
// Created by Song.H.L on 2025/7/5.
//

#ifndef SRC_SA_H
#define SRC_SA_H

#include "instance.h"

void repair(const Instance &inst, Solution &s);

Solution genInitial(const Instance &inst);

double evaluate(const Instance &inst, const Solution &S);

void swapOp(Solution &s);

void insertOp(Solution &s);

void invertOp(Solution &s);

void changeType(Solution &s);

Solution neighbor_4operator(const Instance &inst, const Solution &cur);

void orOptOp(Solution &s, int k);

void crossExchangeOp(Solution &s);

Solution neighbor_6operator(const Instance &inst, const Solution &cur);

RunStat solveInstance(const Instance &inst);

#endif //SRC_SA_H
