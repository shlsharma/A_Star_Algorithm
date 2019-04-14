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

obstacles = ((2,1),(3,1),(4,1),(2,2),(2,3),(2,4),(3,4),(4,4))

state_space = np.zeros((6,6))

queue=np.array([])
initial_state = [3.0,3.0]
goal = [0.0,3.0]
initial_state_g = 0
initial_state_h =  np.linalg.norm(np.asarray(initial_state[0]) - np.asarray(goal[0]))+np.linalg.norm(np.asarray(initial_state[1]) - np.asarray(goal[1]))
initial_state_f = initial_state_g+initial_state_h
initial_set=np.asarray([3.0, 3.0 , initial_state_f, initial_state_g, initial_state_h])
queue = initial_set[np.newaxis]

def g_cost(prev_gcost,states,current_state,f_cost):
    g_cost = np.zeros((len(states),1))
    f_cost = np.asarray(f_cost)
    for i in range(len(states)):
        action = np.linalg.norm(states[i,:]-current_state)
        if action == float(1):
            g_cost[i] = (prev_gcost + 1)
        else:
            g_cost[i] = (prev_gcost + math.sqrt(2)) 
        for j in range(len(f_cost)):
          if (states[i,0],states[i,1]) == (f_cost[j,0],f_cost[j,1]):
              if g_cost[i] > f_cost[j,3]:
                 g_cost[i] = f_cost[j,3]
    return g_cost

def h_cost(states):
    h_cost = np.zeros((len(states),1))
    for i in range(len(states)):
        h_cost[i] = np.linalg.norm(np.asarray(states[i,0])-np.asarray(goal[0]))+np.linalg.norm(np.asarray(states[i,1])-np.asarray(goal[1]))
    return h_cost

def f_cost(states,g_cost,h_cost):
    f_cost = np.zeros((len(states),1))
    for i in range(len(states)):
        f_cost[i] = g_cost[i] + h_cost[i]
    return np.concatenate((states, f_cost,g_cost,h_cost), axis=1)

def state_finder(current_state,obstacles,state_space):
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
    global queue
    #print('q:',queue)
    queue=np.append(queue,f_cost,axis=0)
    i=0
    while (i < (len((queue[:,2])))):
        if [queue[i,0],queue[i,1]] in (closed_states):
           queue = np.delete(queue,i,0)
           i=0
        else:
          i=i+1
    for i in range(len(queue[:,2])):
        if queue[i,2] == np.min(queue[:,2]):
            closed_states.append(list(queue[i,0:2]))
            #print('chossen:', queue[i,:])
            return queue[i,:]

current_state = initial_state
prev_g_cost = initial_state_g
save = []
closed_states = []
closed_states.append(current_state)
f_cos = []
while (current_state != goal):
    states = state_finder(current_state,obstacles,state_space)
    g_cos = g_cost(prev_g_cost,states,current_state,f_cos)
    h_cos = h_cost(states)
    f_cos = f_cost(states,g_cos,h_cos)
    #print('f:',f_cos)
    f = priority_queue(f_cos,closed_states)
    #print('ff:',f)
    current_state = list(f[: 2])
    save.append(current_state)
    prev_g_cost = f[3]
save.insert(0, initial_state)
print(save)

def back_track(save,initial_state):
  end = len(save)-1
  save_path=[]
  save = np.asarray(save)
  save_path.append(save[end,:])
  save_path = np.asarray(save_path)
  print(save_path.shape)
  print(save[1,:].shape)
  i=end
  j=i-1
  while(i>=0):
    while(j>=0):
      print('j value:',j)
      if abs(math.sqrt(2)-(np.linalg.norm(save[i,:]-save[j,:]))) < 0.00001 or abs(np.linalg.norm(save[i,:]-save[j,:])) == 1: 
        get =j
      j=j-1
    #print('i,get:',i,get)  
    #print('for i:',i)
    save_path= np.append(save[get,:][np.newaxis],save_path,axis=0)
    i=get
    j=i-1
    if get ==0:
      break
  print (save_path)
  return save_path

save = back_track(save,initial_state)

obstacles = np.asarray(obstacles)
initial_state = np.asarray(initial_state)
goal = np.asarray(goal)
save = np.asarray(save)
plt.plot(obstacles[:,0]+0.5,obstacles[:,1]+0.5, 'rs',markersize=35)
plt.plot(initial_state[0]+0.5,initial_state[1]+0.5, 'bo',markersize=20)
plt.plot(goal[0]+0.5,goal[1]+0.5, 'go',markersize=20)
plt.plot(save[:,0]+0.5,save[:,1]+0.5, 'k--')
plt.axis([0, len(state_space[0]), 0, len(state_space[1])])
plt.grid(True)
plt.axes().set_aspect('equal')
plt.show()
