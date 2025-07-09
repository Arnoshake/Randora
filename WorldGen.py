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
    print(vmin, " ",vmax)
    bounds = [vmin, (vmin + (vmin*0.4)), (vmin + (vmin*0.75)), (vmin + (vmin*0.85)), vmax]  # strictly increasing!

    # Define matching colors
    colors = [
        "#0000cc",   # Blue for < 0
        "#228B22",   # Green
        "#8B4513",   # Brown
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


def let_there_be_light(size, WORLD_SEED): #create land noise
    scale = .01
    # scale = 5.0 / size
    world_map = [ [ 0 for _ in range(size)] for _ in range(size)]
    rng = random.Random(WORLD_SEED)
    row_offset = rng.uniform(0, 1000)
    col_offset = rng.uniform(0, 1000)
    for row in range(size):
        for col in range(size):
            
            x = (col + col_offset) * scale
            y = (row + row_offset) * scale
            val = pnoise2 ( x,y,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED)
            val = ((val + 1)/2) #normalize btw [0,1]

            world_map[row][col] = val

            col_offset += scale
        row_offset +=scale
    
    min_val = min(min(row) for row in world_map)
    max_val = max(max(row) for row in world_map)
    # print (f"MIN: {min_val} MAX: {max_val}")

        
    
    return world_map
def create_mtn_range(elevation_map,point1,point2,peak_height, range_width):
    size = len(elevation_map[0])

    colPoint1, rowPoint1 = point1 
    colPoint2, rowPoint2 = point2 

    num_steps_in_line = max(abs(colPoint2 - colPoint1), abs(rowPoint2 - rowPoint1))

    for step in range(num_steps_in_line):
        t = step / num_steps_in_line #interpolation factor
        interpolated_col_value = colPoint1 * (1 - t) + colPoint2 * t
        interpolated_row_value = rowPoint1 * (1 - t) + rowPoint2 * t

        jitter_strength = 5.0
        jitter_row = noise.pnoise1(step * 0.1) * jitter_strength
        jitter_col = noise.pnoise1((step+1000) * 0.1) * jitter_strength

        interpolated_col_value += jitter_col
        interpolated_row_value += jitter_row
        for col_offset in range(-range_width,range_width +1 ):
            for row_offset in range(-range_width,range_width +1 ):
                row_location = int(interpolated_row_value + row_offset)
                col_location = int(interpolated_col_value + col_offset)

                if 0 <= row_location < size and 0 <= col_location < size:
                    distance = math.hypot(col_location - interpolated_col_value, row_location - interpolated_row_value)
                    # fall_off = max(0,1-(distance / range_width)) #LINEAR
                    fall_off = max(0, 1 - (distance / range_width))
                    # Bias the elevation falloff in a preferred direction (e.g., more to the north)
                    dy = row_location - interpolated_row_value
                    directional_bias = 1.0 - 0.3 * (dy / range_width)  # tweak 0.3 as needed

                    # Combine falloffs
                    combined_falloff = fall_off * directional_bias
                    # if elevation_map[row_location][col_location] < .5:
                    elevation_map[row_location][col_location] = max(elevation_map[row_location][col_location], peak_height * combined_falloff)

    return 
def create_island(elevation_map, center_point, peak_height, range_width):
    size = len(elevation_map[0])
    cx, cy = center_point

    for dy in range(-range_width, range_width + 1):
        for dx in range(-range_width, range_width + 1):
            x = cx + dx
            y = cy + dy

            if 0 <= x < size and 0 <= y < size:
                distance = math.hypot(dx, dy)
                falloff = max(0, 1 - (distance / range_width)) ** 0.5  
                elevation_map[y][x] = max(elevation_map[y][x], peak_height * falloff)
def create_land(world_map,WORLD_HEIGHT):
    random_amount_of_iterations = random.randint(5,20)
    center_point = (int(len(world_map[0])/2),int(len(world_map[0])/2))
    create_island(world_map,center_point,WORLD_HEIGHT/2,int(len(world_map[0])/2))
    for creations in range(random_amount_of_iterations):
        random_height_value = random.randint(30,60)/100.0
        random_x =random.randint(0,WORLD_SIZE)
        random_y =random.randint(0,WORLD_SIZE)
        random_point1 = (random_x,random_y)
        create_island(world_map,random_point1,random_height_value,25)

    random_amount_of_iterations = random.randint(1,4)
    for creations in range(random_amount_of_iterations):
        random_height_value = random.randint(90,100)/100.0
        random_x =random.randint(0,WORLD_SIZE)
        random_y =random.randint(0,WORLD_SIZE)
        random_point1 = (random_x,random_y)
        random_x =random.randint(0,WORLD_SIZE)
        random_y =random.randint(0,WORLD_SIZE)
        random_point2 = (random_x,random_y)
        create_mtn_range(world_map,random_point1,random_point2,random_height_value,5)







# Testing World Seed Generation

# seedAsString = input("Enter a World Seed: ")

seedAsString = create_seed()
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)

print(f"Seed: {seedAsString} ({WORLD_SEED})")
print(random.randint(0, 100))  # Will always be the same for "bananas"

WORLD_SIZE = 100
WORLD_HEIGHT = 1.0
world_map = let_there_be_light(WORLD_SIZE,WORLD_SEED % 256) #currently making smaller for pnoise to handle ... 100,000 unique worlds 
# create_land(world_map,WORLD_HEIGHT)
# water_function(world_map,WORLD_HEIGHT,0.55)
display_world_GUI(world_map,seedAsString)
# create_land(world_map,WORLD_HEIGHT)










write_world_to_file(seedAsString,world_map,"mapSymbol","symbol")
write_world_to_file(seedAsString,world_map,"mapValue","value")


# COLOR PIXELS MAP
# display_world_GUI(world_map)

