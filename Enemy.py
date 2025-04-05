import random
import turtle


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

class Enemy(Sprite):
	def __init__(self, spriteshape, color, startx, starty):
		Sprite.__init__(self, spriteshape, color, startx, starty)
		self.speed = 6
		self.setheading(random.randint(0,360))
