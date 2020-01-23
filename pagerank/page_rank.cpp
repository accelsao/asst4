#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

using namespace std;

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

    double* tmp = (double*)malloc(sizeof(double) * numNodes);

    bool converged = false;
    while(!converged){
        double sum = 0.0;
        #pragma omp parallel for default(shared) reduction(+: sum)
        for(int i=0; i<numNodes; i++){
            tmp[i] = 0.0;
            if(outgoing_size(g, i) == 0){
                sum = sum + damping * solution[i] / numNodes;
            }
        }
        #pragma omp parallel for
        for(int i=0; i<numNodes; i++){
            const Vertex* _start = incoming_begin(g, i);
            const Vertex* _end = incoming_end(g, i);
            for(const Vertex* v=_start; v!=_end; v++){
                assert(*v < numNodes);
                tmp[i] += solution[*v] / outgoing_size(g, *v);
            }
            tmp[i] = damping * tmp[i] + (1.0 - damping) / numNodes;
            tmp[i] += sum;
        }
        double dif = 0;
        #pragma omp parallel for default(shared) reduction (+:dif)
        for(int i=0; i<numNodes; i++){
            dif = dif + abs(tmp[i] - solution[i]);
        }
        converged = dif < convergence;

        // TODO: swap(solution, tmp) get segment fault , dont know why

        for(int i=0;i<numNodes;i++){
            solution[i] = tmp[i];
        }
    }
    free(tmp);

}
