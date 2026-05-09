import math
MID = 96//2

def point(degrees):
    rad = math.radians(degrees)
    cx = MID+10
    px, py = cx, MID-12
    temp_x = px - cx
    temp_y = py - MID
    rotated_x = temp_x * math.cos(rad) - temp_y * math.sin(rad)
    rotated_y = temp_x * math.sin(rad) + temp_y * math.cos(rad)
    final_x = rotated_x + cx
    final_y = rotated_y + MID
    return round(final_x), round(final_y)

PIXELS = [
    point(-0),
    point(-30),
    point(-60),
    point(-90),
    point(-120),
    point(-150),
    point(-180),
]