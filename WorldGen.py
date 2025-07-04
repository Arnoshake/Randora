import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
import math
from noise import pnoise2
#Array Visualization
import matplotlib.pyplot as plt
import numpy as np
print(noise.pnoise2(0.5, 0.5))
def seed_from_string(s):
    # Convert the string into a 32-bit integer using SHA-256
    # whatever string inputted will always return the same randomized seed
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)
    # USAGE:
    #   seed = seed_from_string("BananasApplesHotDogWatermelon")
    #   random.seed(seed)
    #   print(random.randint(0, 100)) WILL ALWAYS RETURN THE SAME NUMBER PER STRING


def let_there_be_light(size, WORLD_SEED):
    scale = 5.0 / size
    world_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    
    rng = random.Random(WORLD_SEED)
    row_offset = rng.uniform(0, 1000)
    for row in range(size):
        for col in range(size):
            col_offset = rng.uniform(0, 1000)
            x = (col + col_offset) * scale
            y = (row + row_offset) * scale
            val = pnoise2 ( x,y,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED)
            val = ((val + 1)/2) #normalize btw [0,1]

            if (val < 0.5): #flatten bell curve
                val = 0.5 * math.sqrt(2 * val)
            else:
                val = 1 - 0.5 * math.sqrt(2 * (1 - val))
            world_map[row][col] = val

            col_offset += scale
        row_offset +=scale
    
    min_val = min(min(row) for row in world_map)
    max_val = max(max(row) for row in world_map)
    print (f"MIN: {min_val} MAX: {max_val}")

        
    
    return world_map
def write_world_to_file(seed_as_string,world_map,file_name,print_type):
    
    
    #print_type = "value" or "symbol"

    min_val = min(min(row) for row in world_map)
    max_val = max(max(row) for row in world_map)
    
    height_range = (max_val - min_val)
    water_threshold = height_range * 0.1 + min_val
    grass_threshold = height_range * 0.5 + min_val
    forest_threshold = height_range * 0.9 + min_val
    mountain_threshold = height_range + min_val
    
    terrain_symbols = {
    "water": "~",
    "grass": "#",
    "forest":"#",
    "mountain": "^"
    }
    file = open(file_name + ".txt","w")
    
    WORLD_SEED = seed_from_string(seed_as_string)

    file.write(f"Seed: {seedAsString} ({WORLD_SEED})\n")
    file.write (f"MIN: {min_val} MAX: {max_val}\nWATER: {water_threshold} GRASS: {grass_threshold} FOREST: {forest_threshold} MTN: {mountain_threshold}\n")
    size = len(world_map[0])
    for row in range(size):
        for col in range(size):
            
            if (print_type == "value"):
                file.write(f"{world_map[row][col]:.2f} ")
                continue
            if (print_type == "symbol"):
                if (world_map[row][col] < water_threshold):
                    file.write (terrain_symbols["water"])
                    continue
                elif (world_map[row][col] < grass_threshold):
                    file.write (terrain_symbols["grass"])
                    continue
                elif (world_map[row][col] < forest_threshold):
                    file.write (terrain_symbols["forest"])
                    continue
                elif (world_map[row][col] < mountain_threshold):
                    file.write (terrain_symbols["mountain"])
                    continue
                else:
                    file.write ("z")

            
        file.write("\n")
    file.close()
    return
def display_world_altitude(world_map):
    terrain_symbols = {
    "water": "~",
    "mountain": "^",
    "grass": "-",
    "forest":"#"
    }
    height_symbol_dict = {}
    for i in range(10):
        key = float(i)/10
        height_symbol_dict[key] = i
    
    size = len(world_map[0])
    for row in range(size):
        for col in range(size):
            print(f"{world_map[row][col]:.9f} ", end="")
            # if (world_map[row][col] < 0.3):
            #     print (terrain_symbols["water"],end="")
            #     continue
            # elif (world_map[row][col] < 0.4):
            #     print (terrain_symbols["grass"],end="")
            #     continue
            # elif (world_map[row][col] < 0.6):
            #     print (terrain_symbols["forest"],end="")
            #     continue
            # elif (world_map[row][col] < 1):
            #     print (terrain_symbols["mountain"],end="")
            #     continue
        print("\n")
    return

def generate_world_altitude():
    # Select a random amount (1-10) of starting points
    # At each point, use wavefront Alg to apply a gradient. At each tile, add some noise (removes lessen artificial look)
    #       Draw ridges or fault lines between random points (Bresenham’s line algorithm or linear interpolation) --> will also apply to rivers
    #       Bias ripple direction (e.g., ripple more in one axis)



    return
    
def fill_world_water():
    # Based on altitude, fill water (creates oceans and lakes)
    return



# Testing World Seed Generation

seedAsString = input("Enter a World Seed: ")
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)

print(f"Seed: {seedAsString} ({WORLD_SEED})")
print(random.randint(0, 100))  # Will always be the same for "bananas"

world_map = let_there_be_light(30,WORLD_SEED % 256) #currently making smaller for pnoise to handle ... 100,000 unique worlds 
write_world_to_file(seedAsString,world_map,"mapSymbol","symbol")
write_world_to_file(seedAsString,world_map,"mapValue","value")


# COLOR PIXELS MAP
data =world_map
plt.imshow(data, cmap="terrain", interpolation='nearest')
plt.colorbar()
plt.show()
# display_world_altitude(world_map)
