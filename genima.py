import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import os
import imageio


# =============================================================================
#                                   FUNCTIONS
# =============================================================================

def make_gif(gif_num,png_dir='out/',gif_dir='gif/'):
  '''
  Creates a gif from output gen images
  '''
  images = []
  
  for file_name in sorted(os.listdir(png_dir)):
      if file_name.endswith('.png'):
          file_path = os.path.join(png_dir, file_name)
          images.append(imageio.imread(file_path))
  imageio.mimsave(gif_dir+str(gif_num)+'.gif', images, duration=0.01)

def create_cromo(low=0,high=255,shape=[32,32]):
    '''
    Create random chromosome with values between 0 and 255
    '''
    return np.random.randint(low,high,shape,dtype=np.uint8)
    

def fitness(og,cromo):
    return np.sum(cv2.absdiff(og,cromo))
    


def create_pop(n,low=0,high=255,shape=[32,32]):
  population = [create_cromo(low,high,shape) for _ in range(n)]
  return population


def select(population,n,og):
  #Generate fitness value list
  fit_list = [fitness(og,g.ravel()) for g in population]
  
  #Combine and sort
  order = [x for y, x in sorted(zip(fit_list, population),
                                key= lambda y: y[0],
                                reverse=False)]
  
  #Return n best
  return order[:n]

def recombine_onepoint(parent1,parent2,p):
  #Check probability
  if random.random() <= p:
    #Define crossover point
    punto = random.randint(0,len(parent1))
    
    #Generate children
    child1 = np.append(parent1[:punto],parent2[punto:])
    child2 = np.append(parent2[:punto], parent1[punto:])

  else:
    #If it doesn't pass prob check, generate clones
    child1 = parent1.copy()
    child2 = parent2.copy()

  return (child1,child2)

def recombine_twopoint(parent1,parent2,p):
  #Check probability
  if random.random() <= p:
    #Define crossover point 1
    
    #Points (each at max 1/3rd and 2/3rds respectively)
    p1 = random.randint(0,len(parent1)//3)
    p2 = random.randint(p1,(len(parent1)//3)*2)
    
    #Generate children
    child1 = np.append(parent1[:p1],np.append(parent2[p1:p2],parent1[p2:]))
    child2 = np.append(parent2[:p1],np.append(parent1[p1:p2],parent2[p2:]))

  else:
    #If it doesn't pass prob check, generate clones
    child1 = parent1.copy()
    child2 = parent2.copy()

  return (child1,child2)


def mutate(cromo,p1,p2,low=0,high=255,mutsize=100):
  '''
  p1 = probability of mutation on chromosome
  p2 = probability of mutation on each gene
  '''
  
  #Copy list
  crom = cromo.copy()

  #Probab of chromosome mutatio
  if random.random() <= p1:


    #Iterate each gen
    for i in range(len(crom)):
      #Probability of mutating each gene
      if random.random() <= p2:
        operation = random.randint(0,1)
        mod = random.randint(0,mutsize)

        #Define sum or sustraction
        if operation == 0:
            #Check if it doesnt go out of 0-255 range
          while crom[i]-mod < 0:
            mod = random.randint(0,mutsize)
          crom[i] -= mod
        else:
          while crom[i]+mod > 255:
            mod = random.randint(0,mutsize)
          crom[i] += mod

  return crom
    


#Main function of genetic algorithm
def genalg(N,original,low,high,p,p1,p2,seed,K,top,maxgens=100,mutsize=100,
           source=None):
  '''
  Main function
  
  Inputs:
  N = population size
  original = Original (destination) image
  low,high = min, max values
  p = probability of parent recombination
  p1 = probability of chromosome level mutation
  p2 = probability of individual gene mutation
  seed = seed for random functions
  K = amount of parents to select by K-way tournament
  top = propagate the best -top- chromosomes to next gen
  mutsize = size of random per-pixel mutations
  source = if none, start from random generation. 
             ff image, start from source image
  
  Returns:
  List with latest generation's chromosomes
  '''
  
  #Original image to array
  og = original.ravel()

  #Set seeds
  random.seed(seed)
  np.random.seed(seed)

  #Generate original population
  #(random if no source image)
  if source is None:
      pop = create_pop(N,low,high,original.shape)
  else:
      src = source.ravel()
      pop = [src for _ in range(N)]
  

  #Gens and best fitness counters
  gens = 0
  best = 1000000

  #Lists for statistics
  bestlist = []
  avglist = []
  minlist = []

  #Main loop

  while best > 0 and gens < maxgens:
    #Sort gen by fitness
    fit = select(pop,N,og)

    #Statistics
    best = fitness(og,fit[0].ravel())
    bestlist.append(best)

    #TO DO
    avg = 0
    avglist.append(avg)

    #Min is actually max, or worst chromosome
    min = fitness(og,fit[-1].ravel())
    minlist.append(min)



    #Offspring list
    offspring = []

    #N children generated by K-way tournament
    for _ in range(N//2):
      #Parent 1
      indices = np.random.choice(N,K)
      chose = [pop[i] for i in indices]
      parent1 = select(chose,1,og)[0]

      #Parent 2
      indices = np.random.choice(N,K)
      chose = [pop[i] for i in indices]
      parent2 = select(chose,1,og)[0]

      #Recombine and add children
      c1,c2 = recombine_twopoint(parent1.ravel(),parent2.ravel(),p)
      offspring.append(c1)
      offspring.append(c2)


    #Mutate
    mutated = [mutate(c,p1,p2,low,high,mutsize) for c in offspring]

    #Sort
    ord = select(mutated,N,og)

    #Replace x worst with x best of each gen
    new = ord[:N-top] + fit[:top]

    #Replace pop
    pop = new.copy()

    gens += 1

    #Print stats
    #Output best chromosome each 500 gens
    if gens % 300 == 0:
      print('Gen n %d // Best %d // avg %d // min %d' % (gens,best,avg,min))
      image = np.reshape(ord[0],original.shape)
      cv2.imwrite('out/'+"{:.6f}".format(gens/1000000)+'.png',image)


  #Plot evolution
  plt.plot(bestlist)
  plt.plot(avglist)
  plt.plot(minlist)

  plt.show()

  #Return final population
  return pop




# =============================================================================
#                               MAIN
# =============================================================================

img = cv2.imread('b.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

src = cv2.imread('a.png')
#src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

final = genalg(N=20,
               original = img,
               low=0,
               high=255,
               p=0.9,
               p1=0.8,
               p2=0.01,
               seed=1313,
               K=5,
               top=4,
               maxgens=200000,
               mutsize=10,
               source=src)

fbest = select(final,1,img.ravel())
fbest = np.reshape(fbest[0],img.shape)

cv2.imwrite('best.png',fbest)
make_gif(22)
