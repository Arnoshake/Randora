import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
import math
from noise import pnoise2
#Array Visualization
import matplotlib.pyplot as plt
import numpy as np
#VORONOI SEEDING
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.spatial import Voronoi,voronoi_plot_2d

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

    file.write(f"Seed: {seed_as_string} ({WORLD_SEED})\n")
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
    size = len(world_map[0])
    vmin = np.min(world_map)
    vmax = np.max(world_map)
    vrange = vmax - vmin
    print(vmin, " ",vmax)
    water_threshold = (vmin + (vrange*0.4))
    sand_threshold = (vmin + (vrange*0.44))
    grass_threshold = (vmin + (vrange*0.80))
    forest_theshold = (vmin + (vrange*0.90))
    lower_Mtn_threshold = (vmin + (vrange*0.95))

    bounds = [vmin, water_threshold, sand_threshold, grass_threshold,lower_Mtn_threshold, vmax]  # strictly increasing!

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

    # Plot - OLD
    # img = plt.imshow(clamped_map, cmap=cmap, norm=norm, interpolation="nearest")
    # cbar = plt.colorbar(img, ticks=bounds)
    # cbar.set_label("Elevation")

    
    fig, ax = plt.subplots() #fig = orig map, ax = cropped, viewing portion
    img = ax.imshow(clamped_map, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(img, ax=ax, ticks=bounds)
    cbar.set_label("Elevation")
    # Initial camera position and zoom
    zoom_radius = 30
    center_x, center_y = size // 2, size // 2

    def update_view():
        ax.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
        ax.set_ylim(center_y + zoom_radius, center_y - zoom_radius)  # y-axis inverted
        fig.canvas.draw_idle()
    def on_key(event):
        nonlocal center_x, center_y, zoom_radius
        step = 5
        zoom_step = 5

        if event.key == 'left':
            center_x -= step
        elif event.key == 'right':
            center_x += step
        elif event.key == 'up':
            center_y -= step
        elif event.key == 'down':
            center_y += step
        elif event.key == '+' or event.key == '=': #done for mac users
            zoom_radius = max(5, zoom_radius - zoom_step)
        elif event.key == '-':
            zoom_radius = min(size // 2, zoom_radius + zoom_step)

        # Clamp to world bounds
        center_x = max(zoom_radius, min(center_x, size - zoom_radius))
        center_y = max(zoom_radius, min(center_y, size - zoom_radius))

        update_view()

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.set_title(f"World: {SEED_AS_STRING}  \nArrows to pan, +/- to zoom")
    update_view()

    try:
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
    except:
        pass
    #plt.show()
    # plt.title(SEED_AS_STRING)
    # plt.show()
    # plt.imshow(world_map, cmap="gist_earth", interpolation='nearest',alpha =1.0,vmin = -1,vmax = 1)
    # plt.colorbar()
    # plt.show()
    return


    

def create_altitude_map(size, WORLD_SEED): #create land noise
    zoom_level = 1.30  # you can tweak this higher/lower
    #lower at higher

    # scale = zoom_level / size
    scale = 0.01
    print(scale)
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
                            base=WORLD_SEED%256)
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
    
    cx, cy = size / 2, size / 2  # center of map 
    height = len(altitude_map)
    for row in range(size):
        for col in range(size):
            altitude = altitude_map[row][col]
            dx = col - cx
            dy = row - cy
            distance = math.sqrt(dx**2 + dy**2)
            max_dist = math.sqrt(2) * (size / 2)
            base_temp = 1 - (distance / max_dist)  # warm in center, cold on edge
            
            x = (col* scale + col_offset) 
            y = (row* scale + row_offset) 
            temp_noise = pnoise2 ( x,y,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED%256)
            
            temperature= base_temp + temp_noise * 0.5 #add noise to make more natural
            
            temperature -= (altitude ** 1.25) * 0.2 #Decreases temp as altitude increases
            
            temperature+=0.25#linear, across the board temp boost
            cluster_noise = pnoise2(col * 0.01, row * 0.01, octaves=1, base=WORLD_SEED % 256)
            temperature += cluster_noise * 0.1  # mild warping
            temperature = max(0, min(1, temperature))

            
            temp_map[row][col] = temperature

    

        
    
    return temp_map

def Voronoi_seeding(size):
    

    vor_points = np.random.rand(10,2) * size
    
    
   
    vor_points = np.array(vor_points,dtype=float) #converting to NumPy array (for [:,0] & [:,1])
    plt.scatter(vor_points[:,0],vor_points[:,1])
    if len(vor_points) == 0: print("No seed points generated")
    
    if len(vor_points > 3):
        vor_object = Voronoi(vor_points)
        vor_vertices = vor_object.vertices
        vor_regions = vor_object.regions
        voronoi_plot_2d(vor_object,show_vertices=False,line_colors = 'blue')
    plt.title("Voronoi Seeding")
    # plt.legend()
   # plt.show()

    vor_ID_map = np.zeros((size, size), dtype=int)

    for col in range(size):
        for row in range(size):
            min_dist = float('inf')
            min_ID = 0
            for ID,seed in enumerate(vor_points):
                sx, sy = seed[0],seed[1]
                dist = ((sy - row)**2) + ((sx-col)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_ID = ID
            vor_ID_map[row][col] = min_ID

    plt.figure("Region")
    plt.imshow(vor_ID_map,cmap="gray",label="Regions")
    plt.colorbar()
    plt.title("Generated Voronoi Regions")
    #plt.show()
    



    return vor_points,vor_ID_map
def identify_border_cells(vor_regions,size):
    is_vor_border = np.zeros((size, size), dtype=int)

    for row in range(size):
        for col in range(size):
            if (row > 0 and row < (size - 1)): # ensures within vert bounds
                if (vor_regions[row][col] != vor_regions[row+1][col]) or (vor_regions[row][col] != vor_regions[row-1][col]): is_vor_border[row][col] = 1
            if (col > 0 and col < (size - 1)):
                if (vor_regions[row][col] != vor_regions[row][col-1]) or (vor_regions[row][col] != vor_regions[row][col+1]): is_vor_border[row][col] = 1

    plt.figure("Boundaries")
    plt.imshow(is_vor_border,cmap="gray",label="Boundaries")
    plt.colorbar()
    plt.title("Generated Voronoi Edges")
    #plt.show()
    return is_vor_border
def create_tectonic_plates(vor_ID,vor_regions,size,WORLD_SEED):
    #setting random to WORLD_SEED
    rng = np.random.default_rng(WORLD_SEED)

    def create_random_unit_vector():
        theta= math.radians( rng.integers(0,361) )
        x = math.cos( theta)
        y = math.sin(theta)
        return (x,y)
    
    tectonic_plate_dict = {}
    for regions_ID in range(len(vor_ID)):
        if (rng.integers(0,101) > 50):
            plate_type = "continental"
            base_elevation = rng.uniform(0.2,0.4)
        else: 
            plate_type = "oceanic"
            base_elevation = rng.uniform(-0.4,-0.2)


        tect_plate_info = {}
        tect_plate_info["drift"] = create_random_unit_vector()
        tect_plate_info["type"] = plate_type
        tect_plate_info["base_elevation"] = base_elevation
        
        tectonic_plate_dict[regions_ID] = tect_plate_info
    print(tectonic_plate_dict)
    return tectonic_plate_dict

def create_fault_map(vor_ID_list,vor_regions_map,tect_plates,size,WORLD_SEED): # 2 = converge, 1 = transform, 0 = passive, -1 = diverge
    vor_obj = Voronoi(vor_ID_list)
    ridge_points = vor_obj.ridge_points
    fault_lines_map = np.zeros((size,size),dtype=int)
    for (p1, p2), vertex_indices in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
        # (p1,p2) = pair of adjaent plates ... vertex_indices = indices of the ends of their shared line
        if -1 not in vertex_indices: #skips infinite lines/borders
            v1 = vor_obj.vertices[ vertex_indices[0] ]
            v2 = vor_obj.vertices[ vertex_indices[1] ]
            def clip_line_to_bounds(x1, y1, x2, y2, width, height):
                    def clamp(val, minval, maxval):
                        return max(minval, min(val, maxval))

                    # Clip both endpoints to the map boundaries
                    x1_clipped = clamp(x1, 0, width - 1)
                    y1_clipped = clamp(y1, 0, height - 1)
                    x2_clipped = clamp(x2, 0, width - 1)
                    y2_clipped = clamp(y2, 0, height - 1)

                    return int(x1_clipped), int(y1_clipped), int(x2_clipped), int(y2_clipped)
            x1C,y1C,x2C,y2C = clip_line_to_bounds(v1[0],v1[1],v2[0],v2[1],size,size)
            v1 = np.array([x1C,y1C])
            v2 = np.array([x2C,y2C])
            
            dist = ( ( (v2[0]-v1[0])**2) + ( (v2[1]-v1[1])**2) ) **0.5
            failsafe= 1e-8
            t = 1 / max(dist,failsafe)
            t_values = np.linspace(0,1,int(dist)+1)
            #interpolation for the fault line
            
            for t in t_values:
                interpol_pt = v1 + t*(v2 - v1)
                
                p1_vector = tect_plates[p1]["drift"]
                p2_vector = tect_plates[p2]["drift"]
                dot_product = np.dot(p1_vector,p2_vector)

                #Creating a unit vector pointing along the fault
                fault_direction = v2 - v1
                fault_direction = (fault_direction) / np.linalg.norm(fault_direction)
                # creating vector perpendicular to faul     n = [-y,x]
                fault_normal = np.array([-fault_direction[1],fault_direction[0]])

                

                p1_normal = np.dot(p1_vector,fault_normal)
                p2_normal = np.dot(p2_vector,fault_normal)
                net_norm = p1_normal + p2_normal

                p1_parallel = np.dot(p1_vector,fault_direction)
                p2_parallel = np.dot(p2_vector,fault_direction)
                parallel_diff = abs(p1_parallel - p2_parallel)

                row = int(np.clip(math.floor(interpol_pt[1]), 0, size - 1))
                col = int(np.clip(math.floor(interpol_pt[0]), 0, size - 1))

                
                if ( 0 <= row < size and 0 <= col < size):

                    if net_norm < -0.5: fault_lines_map[row][col] = 1#strong convergence
                    elif net_norm > 0.5: fault_lines_map[row][col] = -1#strong divergence
                    else: #passive or transform
                        if parallel_diff > 0.5: fault_lines_map[row][col] = 0#Transform     --> DETERMINE LATER WHAT THIS WILL DO
                        else: fault_lines_map[row][col] = 0# passive
                            
    plt.figure("Faults")
    plt.imshow(fault_lines_map,cmap="gray",label="Regions")
    plt.colorbar()
    plt.title("Generated Plates")
    return fault_lines_map
def world_by_plates(vor_ID_list,vor_regions_map,size,WORLD_SEED):
    
    altitude_map = np.zeros((size,size),dtype=float)
    
    def create_plate_adjacency(vor_points):
        vor_object = Voronoi(vor_points)
        ridge_points = vor_object.ridge_points

        adjacency = {}
        for a,b in ridge_points:
            if a not in adjacency:
                adjacency[a] = []
            if b not in adjacency:
                adjacency[b] = []
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency
    adjacency_dict = create_plate_adjacency(vor_ID_list)

    tect_plates = create_tectonic_plates(vor_ID_list,vor_regions_map,size,WORLD_SEED)
    fault_map = create_fault_map(vor_ID_list,vor_regions,tect_plates,size,WORLD_SEED)
    def apply_noise_to_continental_plate(terrain_map,vor_ID_of_interest,vor_regions,size, WORLD_SEED): #create land noise
        #lower at higher

        # scale = zoom_level / size
        scale = 0.01
        world_map = terrain_map
        rng = random.Random(WORLD_SEED)
        row_offset = rng.uniform(0, 50)
        col_offset = rng.uniform(0, 50)
        for row in range(size):
            for col in range(size):
                if vor_regions[row][col] == vor_ID_of_interest:
                    x = (col* scale + col_offset) 
                    y = (row* scale + row_offset) 
                    val = pnoise2 ( x,y,
                                    octaves=4,
                                    persistence=0.5,
                                    lacunarity=2.0,
                                    base=WORLD_SEED%256)
                    # val = ((val + 1)/2) #normalize btw [0,1]

                    world_map[row][col] += val *.4

        return
    

    for rows in range(size): 
        for cols in range(size):
            altitude_map[rows][cols] = tect_plates[ vor_regions[rows][cols] ]["base_elevation"]
    for plate_id,coords in enumerate(vor_ID_list):
        #for continental plate, apply perlin noise to make into land
        if tect_plates[plate_id]["type"] == "continental":
            apply_noise_to_continental_plate(altitude_map,plate_id,vor_regions,size, WORLD_SEED)



    colors = ["#0000FF", "#FFDAB9", "#228B22", "#A9A9A9", "#FFFFFF"]
    bounds = [0.0, 0.2, 0.3, 0.6, 0.9, 1.0]
    plt.figure("World Map")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))
    plt.imshow(altitude_map, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.legend()
    plt.title("Generated World Map By Plates")

    print("Altitude range:", np.min(altitude_map), "→", np.max(altitude_map))
    plt.figure("Debug Unclipped World")
    colors = [
        "#000033",  # deep ocean (< -0.4)
        "#000080",  # mid ocean  (-0.4 to -0.2)
        "#0077be",  # shallow ocean (-0.2 to 0)

        "#f4a261",  # beach/sand (0 to 0.1)
        "#3B9C35",  # grassy green (0.1 to 0.5)
        "#707070",  # mountains (0.5 to 0.7)
        "#ffffff",  # snowy peaks (0.7+)
    ]
    bounds = [-1.0, -0.4, -0.2, 0.0, 0.1, 0.5, 0.7, 1.0]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))
    plt.imshow(altitude_map, cmap=cmap,norm=norm)
    plt.colorbar()
    return altitude_map


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

            if altitude < water_threshold: biome = "Ocean"
            elif altitude < sand_threshold: biome = "Beach"
            #MAIN LAND AREA
        
            elif altitude < grass_threshold:
                if temperature > 0.8: biome = "Grassland"
                elif temperature > 0.75: biome = "Forest"
                elif temperature > 0.65: biome = "Grassland"
                elif temperature > 0.35: biome = "Forest"
                else:   biome = "Tundra"
            #MOUNTAINS
            elif altitude < lower_Mtn_threshold:
                biome = "Mountain"
            else:
                biome = "Snowy Peaks" if temperature > 0.2 else "Glacier"
            biome_map[row][col] = biome
    return biome_map
def display_biomes_GUI(biome_map,SEED_AS_STRING):
    biome_colors = {
    "Ocean": "#0077b6",
    "Beach": "#f4a261",
    "Grassland": "#90be6d",
    "Tundra": "#c0d6c1",
    "Forest": "#2a9d8f",
    "Mountain": "#8d99ae",
    "Glacier": "#bde0fe",
    "Snowy Peaks": "#F4E9E9"
}

    biome_list = list(biome_colors)   
    color_list = list(biome_colors.values())

    biome_to_integer = {biome: i for i, biome in enumerate(biome_list)}
    int_map = np.array([[biome_to_integer[biome] for biome in row] for row in biome_map]) #using integer casted values, recreate biome map with integer ID instead of String
    
    #assigns colors to the boundary values
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(boundaries=np.arange(len(biome_list)+1)-0.5, ncolors=len(biome_list))

    plt.figure(figsize=(10, 10))
    img = plt.imshow(int_map, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(img, ticks=np.arange(len(biome_list)))
    cbar.ax.set_yticklabels(biome_list)
    cbar.set_label("Biome Type")
    plt.title("Biome Map")
    plt.axis("off")
   # plt.show()
    return










# Testing World Seed Generation

# seedAsString = input("Enter a World Seed: ")
def main():
    print("Starting Program...")

    #SEED GENERATION
    seedAsString = create_seed()
    WORLD_SEED = seed_from_string(seedAsString)
    random.seed(WORLD_SEED)

    print(f"Seed: {seedAsString} ({WORLD_SEED})")
    print(random.randint(0, 100))  # Will always be the same for "bananas"
    #WORLD GENERATION
    WORLD_SIZE = 512*2
    WORLD_HEIGHT = 1.0
    alt_map = create_altitude_map(WORLD_SIZE,WORLD_SEED % 256) #currently making smaller for pnoise to handle ... 100,000 unique worlds 
    temp_map = create_temp_map(WORLD_SIZE,alt_map,WORLD_SEED)
    biome_map = assign_biomes(WORLD_SIZE,alt_map,temp_map,WORLD_SEED)
    #DISPLAY
    display_world_GUI(alt_map,seedAsString)
    # display_biomes_GUI(biome_map,seedAsString)

    write_world_to_file(seedAsString,alt_map,"mapSymbol","symbol")
    write_world_to_file(seedAsString,temp_map,"mapValue","value")


    # COLOR PIXELS MAP
    # display_world_GUI(world_map)

print("Starting Program...")

#SEED GENERATION
seedAsString = create_seed()
WORLD_SEED = seed_from_string(seedAsString)
random.seed(WORLD_SEED)
print(f"Seed: {seedAsString} ({WORLD_SEED})")

size = 100

seeds,vor_regions = Voronoi_seeding(size)
identify_border_cells(vor_regions,size)
plates = create_tectonic_plates(seeds,vor_regions,size,WORLD_SEED)
world = world_by_plates(seeds,vor_regions,size,WORLD_SEED)

plt.show()
