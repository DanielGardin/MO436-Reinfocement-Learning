import sys, tkinter

def color(red, green, blue):
    red   = int(red   * 255)
    green = int(green * 255)
    blue  = int(blue  * 255)
    return f"#{red:02x}{green:02x}{blue:02x}"


class Window:
    BACKGROUND_COLOR = color(0,  0, 0)

    WALL_COLOR  = color(0, .2, 1)
    WALL_RADIUS = 0.15

    SCARED_COLOR = color(1,1,1)

    GHOST_COLORS = [
        color(0.9,   0,   0), # Red
        color(  0, 0.3, 0.9), # Blue
        color(.98, .41, .07), # Orange
        color(0.1, .75, 0.7), # Green
        color(  1, 0.6,   0), # Yellow
        color(0.4, .13, .91)  # Purple
    ]
    GHOST_SIZE = 0.65

    PACMAN_COLOR = color(1, 1, .24)
    PACMAN_SIZE  = 0.5

    FOOD_COLOR = color(1, 1, 1)
    FOOD_SIZE  = 0.1

    CAPSULE_COLOR = color(1,1,1)
    CAPSULE_SIZE = 0.25

    def __init__(self,
                 width=640,
                 height=480
                 ):
        self.windows = tkinter.Tk()

        self.canvas  = tkinter.Canvas(self.windows, width=width, height=height)
        self.canvas.pack()
        
        self._corners = [(0,0), (0, height-1), (width-1, height-1), (width-1, 0)]

        self.draw_polygon(self._corners, self.BACKGROUND_COLOR, smooth=False)

        self.canvas.update()


    def draw_polygon(self, 
                     corners,
                     outline_color,
                     fill_color = None, 
                     filled = True,
                     smooth = True,
                     behind = 0,
                     width = 1):
        coords = []

        for corner in corners:
            x, y = corner
            coords.append(x)
            coords.append(y)
        
        if fill_color is None:
            fill_color = outline_color
        
        if not filled:
            fill_color = ""
        
        poly = self.canvas.create_polygon(coords, outline=outline_color, fill=fill_color, \
                                          smooth=smooth, width=width)
    
        if behind > 0:
            self.canvas.tag_lower(poly, behind)

        return poly

    def draw_square(self, pos, side, color, filled=True, )