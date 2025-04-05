def move(self):
    self.fd(self.speed)  # all the objects from the screen that we will create will have this method available
    # so they will be able to move

while True:
    player.move() # forward with players speed
    enemy.move()
    ally.move()