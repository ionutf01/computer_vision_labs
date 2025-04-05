import random
import turtle

turtle.fd(0) # create the screen
turtle.speed(0) # the speed of the animation
turtle.bgcolor("black")
turtle.ht() # hide default object that is being created
turtle.setundobuffer(1) # limits the amount of memory turtle uses
turtle.tracer(1) # speeding up the program


class Sprite(turtle.Turtle): # the objects on the screen are called sprites. turtles = sprites
# we can use the turtle things inside this "container"
    def __init__(self, spriteshape, color, startx, starty):
        turtle.Turtle.__init__(self, shape=spriteshape)
        # every object of our game, we will set it so it can appear as fast as possible
        self.speed(0) # we will use self as the object will be modified
         # is the fastest, it is the animation speed
        self.penup() # configuration, we don't want oyr object to draw something on the screen
        self.color(color) # the color o the object will be the color we define
        self.fd(0) # ne asiguram ca obiectul apare pe ecran
        self.goto(startx, starty) # mutam obiectul in pozitia de start

        self.speed = 1 # the speed of the object

    def move(self):
        self.fd(self.speed) # all the objects from the screen that we will create will have this method available
        # so they will be able to move
        if self.xcor() > 290:
            self.setx(290)
            self.rt(60)
        if self.xcor() < -290:
            self.setx(-290)
            self.rt(60)
        if self.ycor() > 290:
            self.sety(290)
            self.rt(60)
        if self.ycor() < -290:
            self.sety(-290)
            self.rt(60)


    def is_collision(self, other):
        if (self.xcor() >= (other.xcor() - 20)) and \
                (self.xcor() <= (other.xcor() + 20)) and \
                (self.ycor() >= (other.ycor() - 20)) and \
                (self.ycor() <= (other.ycor() + 20)):
            return True
        else:
            return False


class Player(Sprite):
    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)
        self.speed = 1 # default speed
        self.lives = 3 # 3 vieti
         # proprietate unica pentru player, orice player are vieti, dar nu orice obiect de pe ecran are vieti

    def turn_left(self):
        self.lt(45)

    def turn_right(self):
        self.rt(45)

    def accelerate(self):
        self.speed += 1

    def decelerate(self):
        self.speed -= 1

class Enemy(Sprite):
	def __init__(self, spriteshape, color, startx, starty):
		Sprite.__init__(self, spriteshape, color, startx, starty)
		self.speed = 6
		self.setheading(random.randint(0,360))


class Ally(Sprite):
    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)
        self.speed = 8
        self.setheading(random.randint(0, 360))

    def move(self):
        self.fd(self.speed)

        # Boundary detection
        if self.xcor() > 290:
            self.setx(290)
            self.lt(60)

        if self.xcor() < -290:
            self.setx(-290)
            self.lt(60)

        if self.ycor() > 290:
            self.sety(290)
            self.lt(60)

        if self.ycor() < -290:
            self.sety(-290)
            self.lt(60)

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


# create the game objects
game = Game()

# desenam rama
game.draw_border()

# Creating some objects
player = Player("triangle", "white", 0, 0)
enemy = Enemy("circle", "red", -100, 0)
ally = Ally("square", "blue", 0, 0)

turtle.onkey(player.turn_left, "Left")
turtle.onkey(player.turn_right, "Right")
turtle.onkey(player.accelerate, "Up")
turtle.onkey(player.decelerate, "Down")
turtle.listen()

# Main game loop
while True:
    player.move() # forward with players speed
    enemy.move()
    ally.move()

    if player.is_collision(enemy):
        x = random.randint(-250, 250)
        y = random.randint(-250, 250)
        enemy.goto(x, y)

delay = input("Press enter to finish. >")