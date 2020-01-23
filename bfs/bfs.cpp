#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <utility>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

#define DEBUG

using namespace std;

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighbouring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    bool* vis,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];
        int new_frontier_distance = distances[node] + 1;

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        int tmp[end_edge - start_edge];
        int cnt=0;
        // sequential, push the new node to tmp
        for (int neighbour=start_edge; neighbour<end_edge; neighbour++) {
            int outgoing = g->outgoing_edges[neighbour];
            if (distances[outgoing] == NOT_VISITED_MARKER){
                distances[outgoing] = new_frontier_distance;
                tmp[cnt++]=outgoing;
            }
        }
        // push node in tmp to new_frontier at once
        if(cnt){
            int index = __sync_fetch_and_add(&new_frontier->count, cnt);
            for(int j=0;j<cnt;j++){
                new_frontier->vertices[index + j] = tmp[j];
            }
        }
    }
    // bottom-up need to maintain vis, so hybrid method need top-down to maintain vis either.
    for(int i=0;i<new_frontier->count; i++){
        vis[new_frontier->vertices[i]]=true;
    }
}

void bottom_up_step(
    Graph g,
    bool* vis,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int new_dis = distances[frontier->vertices[0]] + 1;
    int nodes_per_thread = 256;

    // What's the difference between ¡§static¡¨ and ¡§dynamic¡¨ schedule in OpenMP? - stackoverflow
    // https://stackoverflow.com/questions/10850155/whats-the-difference-between-static-and-dynamic-schedule-in-openmp

    #pragma omp parallel for schedule(dynamic, 8)
    for(int i=0; i < g->num_nodes; i+=nodes_per_thread){
        int cnt=0;
        int sz = (i + nodes_per_thread >= g->num_nodes) ? (g->num_nodes - i) : nodes_per_thread;
        int tmp[sz];
        for(int j=i; j < i + sz; j++){
            if(!vis[j]){
                int start_edge = g->incoming_starts[j];
                int end_edge = (j == g->num_nodes - 1)
                               ? g->num_edges
                               : g->incoming_starts[j + 1];

                for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    if(vis[incoming]){
                        distances[j] = new_dis;
                        tmp[cnt++] = j;
                        break;
                    }
                }
            }
        }
        // __sync_fetch_and_add, doc: https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Atomic-Builtins.html
        int index = __sync_fetch_and_add(&new_frontier->count, cnt);
        for(int j=0;j<cnt;j++){
            new_frontier->vertices[index + j] = tmp[j];

        }
    }
    for(int i=0;i<new_frontier->count; i++){
        vis[new_frontier->vertices[i]]=true;
    }
}

void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;


    bool* vis = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        vis[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    vis[ROOT_NODE_ID] = true;

    while (frontier->count > 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, vis, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        swap(frontier, new_frontier);

    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
//     CS149 students:
//
//     You will need to implement the "bottom up" BFS here as
//     described in the handout.
//
//     As a result of your code's execution, sol.distances should be
//     correctly populated for all nodes in the graph.
//
//     As was done in the top-down case, you may wish to organize your
//     code by creating subroutine bottom_up_step() that is called in
//     each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    bool* vis = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        vis[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    vis[ROOT_NODE_ID] = true;

    while(frontier->count > 0){
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, vis, frontier, new_frontier, sol->distances);
        swap(frontier, new_frontier);
    }
    free(vis);
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    bool* vis = (bool*) malloc(graph->num_nodes * sizeof(bool));

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        vis[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    vis[ROOT_NODE_ID] = true;

    int left = graph->num_nodes - 1;

    while(frontier->count > 0){
        vertex_set_clear(new_frontier);
        // compare the ratio of count and left is intuitive
        // 0.1 is hyperparameter, I dont know which is optimize
        if(frontier->count < left * 0.1){
            top_down_step(graph, vis, frontier, new_frontier, sol->distances);
        }
        else{
            bottom_up_step(graph, vis, frontier, new_frontier, sol->distances);
        }
        swap(frontier, new_frontier);
        left -= frontier->count;
    }

}
