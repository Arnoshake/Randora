import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
import math
from noise import pnoise2
#Array Visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

#LIST OF WORDS FOR RANDOMIZED SEEDS
from nltk.corpus import words as nltk_words
WORLD_LIST = nltk_words

print(noise.pnoise2(0.5, 0.5))
def seed_from_string(s):
    # Convert the string into a 32-bit integer using SHA-256
    # whatever string inputted will always return the same randomized seed
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)
    # USAGE:
    #   seed = seed_from_string("BananasApplesHotDogWatermelon")
    #   random.seed(seed)
    #   print(random.randint(0, 100)) WILL ALWAYS RETURN THE SAME NUMBER PER STRING
def create_seed():
    seed = ""
    trueRandom = random.SystemRandom()
    random_number = trueRandom.randint(1,10)
    for _ in range(random_number):
        seed += trueRandom.choice(nltk_words.words()).capitalize()
    return seed

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
            
        print("\n")
    return
def display_world_GUI(world_map,SEED_AS_STRING):
    vmin = np.min(world_map)
    vmax = np.max(world_map)
    vrange = vmax - vmin
    print(vmin, " ",vmax)
    water_threshold = (vmin + (vrange*0.4))
    sand_threshold = (vmin + (vrange*0.44))
    grass_threshold = (vmin + (vrange*0.80))
    lower_Mtn_threshold = (vmin + (vrange*0.95))

    bounds = [vmin, water_threshold, sand_threshold, grass_threshold, lower_Mtn_threshold, vmax]  # strictly increasing!

    # Define matching colors
    colors = [
        "#0000cc",   # Blue for < 0
        "#C2B280",   # tan
        "#228B22",   # Green
        "#707070",   # Gray
        "#ffffff"    # White
    ]

    # Create colormap and norm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    # Clamp world_map to valid range for bounds (just in case)
    clamped_map = np.clip(world_map, -1.0, 1.0)

    # Plot
    img = plt.imshow(clamped_map, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(img, ticks=bounds)
    cbar.set_label("Elevation")
    plt.title(SEED_AS_STRING)
    plt.show()
    # plt.imshow(world_map, cmap="gist_earth", interpolation='nearest',alpha =1.0,vmin = -1,vmax = 1)
    # plt.colorbar()
    # plt.show()
    return


def create_altitude_map(size, WORLD_SEED): #create land noise
    zoom_level = 1.30  # you can tweak this higher/lower
    #lower at higher

    scale = zoom_level / size
    
    world_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    rng = random.Random(WORLD_SEED)
    row_offset = rng.uniform(0, 50)
    col_offset = rng.uniform(0, 50)
    for row in range(size):
        for col in range(size):
            
            x = (col* scale + col_offset) 
            y = (row* scale + row_offset) 
            val = pnoise2 ( x,y,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED)
            val = ((val + 1)/2) #normalize btw [0,1]

            world_map[row][col] = val

    
    min_val = min(min(row) for row in world_map)
    max_val = max(max(row) for row in world_map)
    # print (f"MIN: {min_val} MAX: {max_val}")

        
    
    return world_map
def create_temp_map(size, altitude_map,WORLD_SEED): 
    zoom_level = 1.30  # you can tweak this higher/lower
    #lower at higher

    scale = zoom_level / size
    temp_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    rng = random.Random(WORLD_SEED)
    row_offset = rng.uniform(0, 50)
    col_offset = rng.uniform(0, 50)
    
    height = len(world_map)
    for row in range(size):
        for col in range(size):
           
            
            val = 1 - abs((row/height)*2-1) # 0 - 1 radiating from middle to poles
            x = (col* scale + col_offset) 
            y = (row* scale + row_offset) 

            temp_noise = pnoise2 ( x,y,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED)
            
            temp_map[row][col] = val + temp_noise * 0.2 #add noise to make more natural
            temp_map[row][col] = temp_map[row][col] - (altitude_map[row][col])*0.6 #colder at higher altitude
    

        
    
    return temp_map
def assign_biomes(size,altitude_map,temp_map,WORLD_SEED):
    biome_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    altitude_min = np.min(altitude_map)
    altitude_max = np.max(altitude_map)
    altitude_range = altitude_max-altitude_min

    #WATER
    water_threshold = (altitude_min + (altitude_range*0.4))
    sand_threshold = (altitude_min + (altitude_range*0.44))
    grass_threshold = (altitude_min + (altitude_range*0.80))
    lower_Mtn_threshold = (altitude_min + (altitude_range*0.95))
    #PEAKS
    for row in range(size):
        for col in range(size):
            temperature = temp_map[row][col]
            altitude = altitude_map[row][col]

            if altitude < water_threshold:
                if temperature < 0.2:
                    biome = "Frozen Ocean"
                elif temperature < 0.5:
                    biome = "Ocean"
                else:
                    biome = "Tropical Ocean"

            elif altitude < sand_threshold:
                if temperature < 0.3:
                    biome = "Cold Beach"
                else:
                    biome = "Beach"

            elif altitude < grass_threshold:
                if temperature > 0.75:
                    biome = "Savanna"
                elif temperature > 0.5:
                    biome = "Grassland"
                elif temperature > 0.3:
                    biome = "Cold Steppe"
                else:
                    biome = "Tundra"

            elif altitude < lower_Mtn_threshold:
                if temperature > 0.6:
                    biome = "Deciduous Forest"
                elif temperature > 0.4:
                    biome = "Coniferous Forest"
                else:
                    biome = "Taiga"

            else:  
                if temperature > 0.8: biome = "Volcanic Peaks"
                elif temperature > 0.5:
                    biome = "Rocky Peaks"
                elif temperature > 0.2:
                    biome = "Snowy Peaks"
                else:
                    biome = "Glacier"
            biome_map[row][col] = biome
    return biome_map










# Testing World Seed Generation

# seedAsString = input("Enter a World Seed: ")

seedAsString = create_seed()
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)

print(f"Seed: {seedAsString} ({WORLD_SEED})")
print(random.randint(0, 100))  # Will always be the same for "bananas"

WORLD_SIZE = 100
WORLD_HEIGHT = 1.0
world_map = create_altitude_map(WORLD_SIZE,WORLD_SEED % 256) #currently making smaller for pnoise to handle ... 100,000 unique worlds 
temp_map = create_temp_map(WORLD_SIZE,world_map,WORLD_SEED)
# create_land(world_map,WORLD_HEIGHT)
# water_function(world_map,WORLD_HEIGHT,0.55)
display_world_GUI(world_map,seedAsString)
# create_land(world_map,WORLD_HEIGHT)










write_world_to_file(seedAsString,world_map,"mapSymbol","symbol")
write_world_to_file(seedAsString,temp_map,"mapValue","value")


# COLOR PIXELS MAP
# display_world_GUI(world_map)

