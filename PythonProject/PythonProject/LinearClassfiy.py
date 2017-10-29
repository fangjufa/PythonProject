import matplotlib.pyplot as pl
import numpy as np
#draw rectangle
import matplotlib.patches as patch
#to get time
import time
import matplotlib.animation as funcanim

#def compute_score(W,x,b):
#    s0 = W[0]*x[0] + W[1]*x[1] + b[0]
#    s1 = W[2]*x[0] + W[3]*x[1] + b[1]
#    s2 = W[4]*x[0] + W[5]*x[1] + b[2]
#    return [s0,s1,s2]

#随机生成W和b,W大小为3行两列,b的大小为三行一列
W = np.random.rand(3,2) - 0.5
b = np.random.rand(3) - 0.5
#generate 9 dot points
dots = [[0.25, 0.2],[0.4, 0.15],[0.15, 0.4],[-0.2, 0.15],[-0.15, 0.35],[-0.35, 0.1],[0.35, -0.2],[0.25, -0.3],[-0.2, -0.25]];
labels = [0,0,0,1,1,1,2,2,2];

figure = pl.figure()
axes = figure.add_axes([0,0,1,1])

h_x,h_y = [-0.5,0.5],[0,0]
v_x,v_y = [0,0],[-0.5,0.5]

line_colors = ["red","green","blue"]
area_colors = [[1,0.35,0.35],[0.35,1,0.35],[0.35,0.35,1]]

#draw axis lines
axes.plot(h_x,h_y,color = "black")
axes.plot(v_x,v_y,color = "black")

#draw dots
for i in range(9):
    axes.plot(dots[i][0],dots[i][1],'o',color = line_colors[int(i/3)])

num = 50
x = np.linspace(-0.5,0.5,num)
y = np.linspace(-0.5,0.5,num)
xarray = [x,x,x]

#set x,y limitation
axes.set_xlim([-0.5,0.5])
axes.set_ylim([-0.5,0.5])

losses = np.zeros(9)

gradb = np.zeros(3)
gradw = np.zeros([3,2])
learning_rate = 0.05

last_maxs = -9999

width = 1/num + 0.001
height = 1/num + 0.001

def UpdateParameters():
    global gradb,gradw,losses
    losses = np.zeros(9)
    #update W and b
    #calculate gradient of W and b
    for i in range(9):
        li = labels[i]

        import lossfunction as lf
        gw,gb,losses[i] = lf.LossFuntion().WW(W,b,dots[i],li)
        #gw,gb,losses[i] = lf.LossFuntion().SoftMax(W,b,dots[i],li)
        gradw = np.add(gradw , gw)
        gradb = np.add(gradb, gb)

        #ww
        #scores = np.dot(W,dots[i]) + b
        #for j in range(3):
        #    if li == j:
        #        continue
        #    lossi = np.max([0.0, scores[j] - scores[li] + 1.0])
        #    losses[i] += lossi
        #    if lossi > 0:
        #        #calculate gradient
        #        gradb[j] += 1
        #        gradw[j] += dots[i]
        #        gradb[li] -= 1
        #        gradw[li] -= dots[i]
    gradb = gradb /9
    gradw = gradw /9
    losses /= 9

N = 3
lines = [axes.plot([], [])[0] for _ in range(N)]
#draw lines
#cannot be [3],use range(3)
for i in range(3):
    lines[i], = axes.plot(xarray[i],xarray[i],color = line_colors[i])

count = 0
def animate(frame):
    global W,b,lines,last_maxs,count,anim,gradw,gradb
    W += -learning_rate*gradw
    b += -learning_rate*gradb
    y1 = -(b[0]+W[0][0]*x)/W[0][1]
    y2 = -(b[1]+W[1][0]*x)/W[1][1]
    y3 = -(b[2]+W[2][0]*x)/W[2][1]
    yarray = [y1,y2,y3]
    lines[0].set_data(xarray[0],y1)
    lines[1].set_data(xarray[1],y2)
    lines[2].set_data(xarray[2],y3)
    count += 1

    gradb = np.zeros(3)
    gradw = np.zeros([3,2])
    UpdateParameters()

    print("Now loss is %f,and count is %d" %(np.sum(losses),count))
    #if count >= 100:
    #    print("stop animation")
    #    anim.event_source.stop()
    return lines

anim = funcanim.FuncAnimation(figure, animate,
                               interval=20, blit=True)

pl.show()


        #softmax
        #for k in range(3):
        #    
        #    if k == li:
        #        
        #        losses[j] += -np.log(p[k])
        #        #p[k] is the same for W[2*k] & W[2*k + 1] & b,because p[k] = gradW0 * dot0 + gradW1*dot1+b
        #        #here p[k]-1,actually it is p[k] - y[k]
        #        gradw[2*k] += (p[k] - 1)*dots[j][0]
        #        gradw[2*k + 1] += (p[k] - 1)*dots[j][1]
        #        gradb[k] += p[k]-1
        #    else:
        #        gradw[2*k] += p[k]*dots[j][0]
        #        gradw[2*k + 1] += p[k]*dots[j][1]
        #        gradb[k] += p[k]

    #draw rect area
    #for xi in np.nditer(x):
    #    for yj in np.nditer(y):
    #        count += 1
    #        s1 = W[0]*xi + W[1]*yj + b[0]
    #        s2 = W[2]*xi + W[3]*yj + b[1]
    #        s3 = W[4]*xi + W[5]*yj + b[2]
    #        max_s = np.max([s1,s2,s3])
    #        #if max_s == last_maxs:
    #        #    continue
    #        #last_maxs = max_s
    #        if max_s == s1:
    #            #print(0,":",count)
    #            #should add patch to axes
    #            axes.add_patch(patch.Rectangle((xi - width/2,yj - height/2),width,height,facecolor = area_colors[0]))
    #        elif max_s == s2:
    #            #print(1,":",count)
    #            axes.add_patch(patch.Rectangle((xi - width/2,yj - height/2),width,height,facecolor = area_colors[1]))
    #        else:
    #            #print(2,":",count)
    #            axes.add_patch(patch.Rectangle((xi - width/2,yj - height/2),width,height,facecolor = area_colors[2]))