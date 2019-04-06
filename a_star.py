#///////////////////////////////////////////////////////////////////////////////////////
#//Terms of use
#///////////////////////////////////////////////////////////////////////////////////////
#//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#//THE SOFTWARE.
#///////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) 2019, Sahil Sharma (IIT Patna), All rights reserved.

import numpy as np
import math
import matplotlib.pyplot as plt
#Feel free to change the State_space and Obstacles.
obstacles = ((1,1),(1,2),(2,2),(3,2),(5,3),(6,3),(3,4),(2,4),(3,0),(5,1),(0,5),(2,6),(5,6))
state_space = np.zeros((7,7))
initial_state = [6.0,5.0]
goal = [0.0,0.0]
initial_state_g = 0
initial_state_h =  np.linalg.norm(np.asarray(initial_state) - np.asarray(goal))

def g_cost(prev_gcost,states,current_state):
    g_cost = np.zeros((len(states),1))
    for i in range(len(states)):
        action = np.linalg.norm(states[i,:]-current_state)
        if action == float(1):
            g_cost[i] = (prev_gcost + 1)
        else:
            g_cost[i] = (prev_gcost + math.sqrt(2)) 
    return g_cost

def h_cost(states):
    h_cost = np.zeros((len(states),1))
    for i in range(len(states)):
        h_cost[i] = np.linalg.norm(np.asarray(states[i,:])-np.asarray(goal))
    return h_cost

def f_cost(states,g_cost,h_cost):
    f_cost = np.zeros((len(states),1))
    for i in range(len(states)):
        f_cost[i] = g_cost[i] + h_cost[i]
    return np.concatenate((states, f_cost,g_cost,h_cost), axis=1)

def state_finder(current_state,obstacles,state_space):
    world = creating_world(obstacles,state_space)
    current_state = np.asarray(current_state)
    i=current_state[0]
    j=current_state[1]
    all_states = ((i-1,j-1),(i,j-1),(i+1,j-1),(i+1,j),(i+1,j+1),(i,j+1),(i-1,j+1),(i-1,j))
    all_states = np.asarray(all_states)
    i=0
    while (i < len(all_states)):
        if all_states[i,0] > (len(state_space[:,0])-1):
            all_states = np.delete(all_states,i,0)
            i = 0
        else:
            i = i+1
    i=0
    while (i < len(all_states)):
        if all_states[i,1] > (len(state_space[0,:])-1):
            all_states = np.delete(all_states,i,0)
            i = 0
        else:
            i = i+1
    i=0
    while (i < len(all_states[:,0])):
        if (all_states[i,0],all_states[i,1]) in (obstacles):
            all_states = np.delete(all_states,i,0) 
            i=0
        else:
            i=i+1
    return all_states

def priority_queue(f_cost,closed_states):
    print('closed node:',closed_states)
    i=0
    while (i < (len(f_cost[:,2]))):
        if [f_cost[i,0],f_cost[i,1]] in (closed_states):
           f_cost = np.delete(f_cost,i,0)
           i=0
        else:
          i=i+1
    for i in range(len(f_cost[:,2])):
        if f_cost[i,2] == np.min(f_cost[:,2]):
            closed_states.append(list(f_cost[i,0:2]))
            return f_cost[i,:]

current_state = initial_state
prev_g_cost = initial_state_g
save = []
closed_states = []
closed_states.append(current_state)
while (current_state != goal):
    print(current_state)
    save.append(current_state)
    states = state_finder(current_state,obstacles,state_space)
    g_cos = g_cost(prev_g_cost,states,current_state)
    h_cos = h_cost(states)
    f_cos = f_cost(states,g_cos,h_cos)
    f = priority_queue(f_cos,closed_states)
    current_state = list(f[: 2])
    prev_g_cost = f[3]
save.append(goal)
#Printing world
obstacles = np.asarray(obstacles)
initial_state = np.asarray(initial_state)
goal = np.asarray(goal)
save = np.asarray(save)
plt.plot(obstacles[:,0],obstacles[:,1], 'rs',markersize=25)
plt.plot(initial_state[0],initial_state[1], 'bo',markersize=20)
plt.plot(goal[0],goal[1], 'go',markersize=20)
plt.plot(save[:,0],save[:,1], 'k--')
plt.axis([-1, len(state_space[0]), -1, len(state_space[1])])
plt.grid(True)
plt.axes().set_aspect('equal')
plt.show()