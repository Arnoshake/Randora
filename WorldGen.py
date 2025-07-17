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
#FAULT LINE
from skimage.draw import line

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


    

def create_altitude_map(size, plate_map, WORLD_SEED): #create land noise
    zoom_level = 1.30  # you can tweak this higher/lower
    #lower at higher
    rng = random.Random(WORLD_SEED)
    # scale = zoom_level / size
    scale = 0.01
    #ASSIGN PLATES A BIAS AND RUGGEDNESS
    num_plates = np.max(plate_map) + 1
    plate_bias = [rng.uniform(-0.2,0.4) for _ in range(num_plates)]
    plate_ruggedness = [rng.uniform(0.5,1.5) for _ in range(num_plates)]
    
    altitude_map = np.zeros((size,size))
   
    row_offset = rng.uniform(0, 50)
    col_offset = rng.uniform(0, 50)
    for row in range(size):
        for col in range(size):
            #grab the relevant plate values
            plate_id = plate_map[row][col]
            bias = plate_bias[plate_id]
            rugged = plate_ruggedness[plate_id]


            nx = (col* scale + col_offset) 
            ny = (row* scale + row_offset) 
            pval = pnoise2 ( row,col,
                            octaves=4,
                            persistence=0.5,
                            lacunarity=2.0,
                            base=WORLD_SEED%256)
            altitude = pval * rugged + bias
            altitude_map[row][col] = altitude

    
    min_val = min(min(row) for row in altitude_map)
    max_val = max(max(row) for row in altitude_map)
    # print (f"MIN: {min_val} MAX: {max_val}")
    plt.figure("My Map")           # Optional: set figure title
    plt.imshow(altitude_map, cmap='terrain')  # cmap options: 'gray', 'terrain', 'viridis', etc.
    plt.colorbar()                 # Optional: shows a scale bar
    plt.title("Elevation Map")
    plt.axis('off')                # Optional: hide axes ticks

        
    
    return altitude_map

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

#Vor_regions = 2D Arr holding plate identity
def Voronoi_seeding(size,seed_density,WORLD_SEED):
    rng = random.Random(WORLD_SEED)
    num_seeds = int((size**2) * seed_density)
    vor_points = [[rng.uniform(0, size), rng.uniform(0, size)] for _ in range(num_seeds)]

    
    
   
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

    vor_regions = np.zeros((size, size), dtype=int)

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
            vor_regions[row][col] = min_ID

    plt.figure("Region")
    plt.imshow(vor_regions,cmap="gray",label="Regions")
    plt.colorbar()
    plt.title("Generated Voronoi Regions")
    #plt.show()
    



    return vor_points,vor_regions
def identify_border_cells(size, vor_regions,plate_list,vor_seeds): #returns binary value map of borders 
    is_vor_border = np.zeros((size, size), dtype=int)
    def create_adjacency(size, vor_regions):
        adjacency_dict = {}
        for row in range(size):
            for col in range(size):
                center_plate= vor_regions[row][col]
                if center_plate not in adjacency_dict:
                    adjacency_dict[center_plate] = set()
                for dy,dx in [(+1,0),(-1,0),(0,+1),(0,-1)]:
                    ny, nx = dy + row, dx + col
                    if (0<=ny<size) and (0<=nx<size):
                        neighbor_plate = vor_regions[ny][nx]
                        if neighbor_plate != center_plate:
                            adjacency_dict[center_plate].add(neighbor_plate)
        adjacency = {k: sorted(list(v)) for k, v in adjacency_dict.items()}
        return adjacency
    adj_table = create_adjacency(size,vor_regions)

    for row in range(size):
        for col in range(size):
            if (row > 0 and row < (size - 1)): # ensures within vert bounds
                if (vor_regions[row][col] != vor_regions[row+1][col]) or (vor_regions[row][col] != vor_regions[row-1][col]): is_vor_border[row][col] = 1
            if (col > 0 and col < (size - 1)):
                if (vor_regions[row][col] != vor_regions[row][col-1]) or (vor_regions[row][col] != vor_regions[row][col+1]): is_vor_border[row][col] = 1

    #Identify fault type
    fault_id_table = {}
    for plate_id in adj_table:
        drift1 = plate_list[plate_id]["drift"]
        for neighbors in adj_table[plate_id]:
            if plate_id not in fault_id_table:
                fault_id_table[plate_id] = {}
            if neighbors in fault_id_table[plate_id]:
                continue
            drift2 = plate_list[neighbors]["drift"]
            relative_drift = np.array(drift2) - np.array(drift1) 
            c1x,c1y = vor_seeds[plate_id]
            c2x,c2y = vor_seeds[neighbors]
            direction_vector = np.array([c2x - c1x, c2y - c1y])
            norm = np.linalg.norm(direction_vector)
            if norm == 0: continue
            direction_vector = direction_vector / norm
            projected_speed = np.dot(relative_drift,direction_vector) #difference of plates at the angle of the plate collision

            
            if projected_speed > 0.4: #2
                fault_id_table[plate_id][neighbors] = 2 # CONVERGENT
            elif projected_speed < -0.4: #-1
                fault_id_table[plate_id][neighbors] = -2 # DIVERGENT
            else:
                fault_id_table[plate_id][neighbors] = 1 # OTHER... Setting it for recongition/testing

    fault_map = np.zeros((size,size))
    for row in range(size):
        for col in range(size):
            if is_vor_border[row][col] == 1:
                plate1 = vor_regions[row][col]
                for dy,dx in [[+1,0],[-1,0],[0,+1],[0,-1]]:
                    ny,nx = dy + row,dx + col
                    if (0<=ny<size and 0<=nx<size):
                        plate2 = vor_regions[ny][nx]
                        if plate1 != plate2: #FETCH THE IDENTIFICATION
                            fault_map[row][col] = fault_id_table[plate1][plate2]


    plt.figure("Boundaries")
    plt.imshow(is_vor_border,cmap="gray",label="Boundaries")
    plt.colorbar()
    plt.title("Generated Voronoi Edges")
    plt.figure("FaultMap GIIGGLGLLG")
    plt.imshow(fault_map,cmap="gray",label="Boundaries")
    plt.colorbar()
    plt.title("Generated Fault Map")
    plt.show()
    return is_vor_border
def create_tectonic_plates(vor_ID,vor_regions,size,WORLD_SEED): # returns dict of plates
    #setting random to WORLD_SEED
    rng = np.random.default_rng(WORLD_SEED)

    def create_random_unit_vector():
        theta= math.radians( rng.integers(0,361) )
        x = math.cos( theta)
        y = math.sin(theta)
        return (x,y)
    
    tectonic_plate_dict = {}
    for regions_ID in range(len(vor_ID)):
        if (rng.integers(0,101) > 40):
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
def create_fault_map(vor_ID_list, vor_regions_map, tect_plates, size, WORLD_SEED): 
    vor_obj = Voronoi(vor_ID_list)
    fault_lines_map = np.zeros((size, size), dtype=int)

    def clip(val, minval, maxval):
        return max(minval, min(val, maxval))

    for (p1, p2), vertex_indices in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
        if len(vertex_indices) != 2:
            continue

        # Finite ridge
        if -1 not in vertex_indices:
            v1 = vor_obj.vertices[vertex_indices[0]]
            v2 = vor_obj.vertices[vertex_indices[1]]
        else:
            # Infinite ridge: project it out
            finite_idx = vertex_indices[1] if vertex_indices[0] == -1 else vertex_indices[0]
            v1 = vor_obj.vertices[finite_idx]

            # Get perpendicular direction from seed points
            point1 = vor_obj.points[p1]
            point2 = vor_obj.points[p2]
            direction = point2 - point1
            direction = np.array([-direction[1], direction[0]])  # perpendicular

            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm

            v2 = v1 + direction * (size * 2)  # Project far enough to cross map

        # Skip if invalid or collapsed
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
            continue
        if np.linalg.norm(v2 - v1) < 1e-6:
            continue

        # Convert to clipped pixel coordinates
        r0 = int(clip(round(v1[1]), 0, size - 1))
        c0 = int(clip(round(v1[0]), 0, size - 1))
        r1 = int(clip(round(v2[1]), 0, size - 1))
        c1 = int(clip(round(v2[0]), 0, size - 1))

        rr, cc = line(r0, c0, r1, c1)
        fault_lines_map[rr, cc] = 1

    # Display
    plt.figure("Fault Map")
    plt.imshow(fault_lines_map, cmap="gray")
    plt.title("All Fault Lines (with Infinite Edges)")
    plt.colorbar()
    plt.show()

    return fault_lines_map










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

size = 256

seeds,vor_regions = Voronoi_seeding(size,0.00025,WORLD_SEED)

plates = create_tectonic_plates(seeds,vor_regions,size,WORLD_SEED)
identify_border_cells(size,vor_regions,plates,seeds)
altitude = create_altitude_map(size,vor_regions,WORLD_SEED)
plt.show()
