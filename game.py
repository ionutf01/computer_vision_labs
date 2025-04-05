import turtle


class Game():
    def __init__(self):
        self.level = 1
        self.score = 0
        self.state = "playing"
        self.pen = turtle.Turtle()
        self.lives = 3

        # drawing a border
    def draw_border(self):
        self.pen.speed(0) # animation speed
        self.pen.color("white")
        self.pen.pensize(3)
        self.pen.penup()
        self.pen.goto(-300, 300) # obiectele sunt create la coordonatele 0,0
        # noi vrem ca aceasta rama sa inceapa de la pozitia -300, 300
        # starts in the upper left corner, going in the right direction
        self.pen.pendown()
        for side in range(4):
            self.pen.fd(600)
            self.pen.rt(90)

        self.pen.penup()
        self.pen.ht()