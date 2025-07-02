import random
import hashlib #cryptographic hashes (consistency)
import noise #for more realistic generation
print(noise.pnoise2(0.5, 0.5))
def seed_from_string(s):
    # Convert the string into a 32-bit integer using SHA-256
    # whatever string inputted will always return the same randomized seed
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)
    # USAGE:
    #   seed = seed_from_string("BananasApplesHotDogWatermelon")
    #   random.seed(seed)
    #   print(random.randint(0, 100)) WILL ALWAYS RETURN THE SAME NUMBER PER STRING


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
seed = seed_from_string("BananasApplesHotDogWatermelon")
random.seed(seed)

print(f"Seed: {seed}")
print(random.randint(0, 100))  # Will always be the same for "bananas"