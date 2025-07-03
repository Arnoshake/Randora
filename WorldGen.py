import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
import math
from noise import pnoise2
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
    scale = 0.25
    world_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    
    for iterations in range(random.randint(0, 20)):
        for x in range(size):
            for y in range(size):
                world_map[x][y] = pnoise2(x*scale,y*scale,base=WORLD_SEED) + 1 #pnoise2 provides value between interpolation points, straight integers land at 0's, need scale to shift this off
    
    return world_map

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
            if (world_map[row][col] < 0.3):
                print (terrain_symbols["water"],end="")
                continue
            elif (world_map[row][col] < 1):
                print (terrain_symbols["grass"],end="")
                continue
            elif (world_map[row][col] < 1.5):
                print (terrain_symbols["forest"],end="")
                continue
            elif (world_map[row][col] < 2):
                print (terrain_symbols["mountain"],end="")
                continue
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
seedAsString = "BananasApplesHotDogWatermelon"
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)

print(f"Seed: {seedAsString} : {WORLD_SEED}")
print(random.randint(0, 100))  # Will always be the same for "bananas"

world_map = let_there_be_light(20,WORLD_SEED % 100000) #currently making smaller for pnoise to handle ... 100,000 unique worlds 

display_world_altitude(world_map)
