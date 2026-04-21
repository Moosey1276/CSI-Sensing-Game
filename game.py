import pygame
import os
import random
from shared_state import shared_pose, stop_event
import serial
import time
from model_training import train
from model_conversion import keras2engine
from collection_pipeline import collect_data
import pose_recognition
import threading
import queue

UNKNOWN = pygame.image.load(os.path.join("New_assets", "question_mark.png"))
UNKNOWN = pygame.transform.scale(UNKNOWN, (100, 100))
ARROW = pygame.image.load(os.path.join("New_assets", "arrow_pointing_right.png"))
ARROW = pygame.transform.scale(ARROW,(100,100))

STANDING = pygame.image.load(os.path.join("New_assets/Stick_figure", "Stick_standing2.png"))
STANDING_HOLE = pygame.image.load(os.path.join("New_assets/Holes_in_wall", "Standing_hole.png"))

CROUCH = pygame.image.load(os.path.join("New_assets/Stick_figure", "Stick_crouch.png"))
CROUCH_HOLE = pygame.image.load(os.path.join("New_assets/Holes_in_wall", "Crouch_hole2.png"))

X_POSE = pygame.image.load(os.path.join("New_assets/Stick_figure", "Stick_X_pose.png"))
X_POSE_HOLE = pygame.image.load(os.path.join("New_assets/Holes_in_wall", "X_pose_hole2.png"))

SKI_POSE = pygame.image.load(os.path.join("New_assets/Stick_figure", "Stick_ski_pose.png"))
SKI_POSE_HOLE = pygame.image.load(os.path.join("New_assets/Holes_in_wall", "Ski_pose_hole2.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Human:
    X_POS = 80
    Y_POS = 300
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.unknown_pose = STANDING
        self.standing_img = STANDING
        self.crouch_img = CROUCH
        self.x_img = X_POSE
        self.ski_img = SKI_POSE

        self.unknown = True
        self.standing = False
        self.crouch = False
        self.x_pose = False
        self.ski_pose = False

        self.jump_vel = self.JUMP_VEL
        self.image = self.unknown_pose
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.standing:
            self.stand()
        if self.unknown:
            self.unknown_func()
        if self.crouch:
            self.crouch_func()
        if self.x_pose:
            self.x()
        if self.ski_pose:
            self.ski()

        if userInput["standing"] and not self.standing:
            self.unknown = False
            self.standing = True
            self.crouch = False
            self.x_pose = False
            self.ski_pose = False
            self.current_pose = "standing"
        elif userInput["crouching"] and not self.crouch:
            self.unknown = False
            self.standing = False
            self.crouch = True
            self.x_pose = False
            self.ski_pose = False
            self.current_pose = "crouching"
        elif userInput["ski_pose"] and not self.ski_pose:
            self.unknown = False
            self.standing = False
            self.crouch = False
            self.x_pose = False
            self.ski_pose = True
            self.current_pose = "ski_pose"
        elif userInput["x_pose"] and not self.x_pose:
            self.unknown = False
            self.standing = False
            self.crouch = False
            self.x_pose = True
            self.ski_pose = False
            self.current_pose = "x_pose"
        elif not (self.standing or self.crouch or self.x_pose or self.ski_pose):
            self.unknown = True
            self.standing = False
            self.crouch = False
            self.x_pose = False
            self.ski_pose = False
            self.current_pose = "unknown"

    def stand(self):
        self.image = self.standing_img
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def unknown_func(self):
        self.image = self.unknown_pose
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def crouch_func(self):
        self.image = self.crouch_img
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def x(self):
        self.image = self.x_img
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def ski(self):
        self.image = self.ski_img
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Obstacle:
    def __init__(self, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH
        self.checked = False
        self.speed = game_speed

    def update(self):
        self.rect.x -= self.speed
        if self.speed < 0:
            self.speed += 1
            if self.speed == 0:
                self.speed = game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image, self.rect)

class Standing_hole(Obstacle):
    def __init__(self, image):
        super().__init__(image)
        self.rect.y = 30
        self.required_pose = "standing"

class Crouch_hole(Obstacle):
    def __init__(self, image):
        super().__init__(image)
        self.rect.y = -20
        self.required_pose = "crouching"

class X_pose_hole(Obstacle):
    def __init__(self, image):
        super().__init__(image)
        self.rect.y = 25
        self.required_pose = "x_pose"

class Ski_pose_hole(Obstacle):
    def __init__(self, image):
        super().__init__(image)
        self.rect.y = 5
        self.required_pose = "ski_pose"

def run_pose_recognition(serial_port):
    print("Starting pose recognition thread...")
    pose_recognition.main(serial_port)
    print("Pose recognition thread exited!")

def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, gathered
    run = True
    clock = pygame.time.Clock()
    player = Human()
    game_speed = 8
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    gathered = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    pose_dict = {"standing": STANDING, "crouching": CROUCH, "x_pose": X_POSE, "ski_jump": SKI_POSE}

    ser = serial.Serial('COM3', baudrate=921600, timeout=1, rtscts=False, dsrdtr=False)
    print("Connected to", ser.name)
    time.sleep(2)
    start_time = time.time()
    progress = queue.Queue()
    collection_thread = threading.Thread(target=collect_data, args=(ser, progress,))
    collection_thread.start()
    while run:
        font = pygame.font.Font('freesansbold.ttf', 50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                stop_event.set()
                pygame.quit()
        try:
            message = progress.get_nowait()
        except queue.Empty:
            message = None

        if message:
            if message[0] != "quit":
                if len(message) == 2:
                    text = font.render(f"Do pose {message[0]}!", True, (0, 0, 0))
                    textRect = text.get_rect()
                    textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                    loop_text = font.render(message[1], True, (0, 0, 0))
                    loop_textRect = loop_text.get_rect()
                    loop_textRect.center = (50, 50)
                    SCREEN.blit(text, textRect)
                    SCREEN.blit(loop_text, loop_textRect)
                    SCREEN.blit(pose_dict[message[0]], (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 200))
                    pygame.display.update()
                else:
                    text = font.render(f"Change from {message[0]} to {message[1]}!", True, (0, 0, 0))
                    textRect = text.get_rect()
                    textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                    loop_text = font.render(message[2], True, (0, 0, 0))
                    loop_textRect = loop_text.get_rect()
                    loop_textRect.center = (50, 50)
                    SCREEN.blit(text, textRect)
                    SCREEN.blit(loop_text, loop_textRect)
                    SCREEN.blit(pose_dict[message[0]], (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 200))
                    SCREEN.blit(ARROW, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 200))
                    SCREEN.blit(pose_dict[message[1]], (SCREEN_WIDTH // 2 + 140, SCREEN_HEIGHT // 2 - 200))
                    pygame.display.update()
            else:
                run = False

        clock.tick(1)
        SCREEN.fill((255, 255, 255))

    collection_thread.join()
    train()
    keras2engine()
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time} seconds")
    font = pygame.font.Font('freesansbold.ttf', 30)

    t = threading.Thread(target=run_pose_recognition, args=(ser,), daemon=True)
    t.start()
    run = True

    def score():
        global gathered, points, game_speed
        gathered += 0.5

        if gathered % 5 == 0:
            points += 1

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                stop_event.set()

        SCREEN.fill((255, 255, 255))

        current_pose = shared_pose.get_pose()

        # simulate user input flags for compatibility
        userInput = {
            "standing": current_pose == "standing",
            "crouching": current_pose == "crouching",
            "ski_pose": current_pose == "ski_pose",
            "x_pose": current_pose == "x_pose",
        }

        player.update(userInput)
        player.draw(SCREEN)

        if len(obstacles) == 0:
            rand_num = random.randint(0, 3)
            if rand_num == 0:
                obstacles.append(Standing_hole(STANDING_HOLE))
            elif rand_num == 1:
                obstacles.append(Crouch_hole(CROUCH_HOLE))
            elif rand_num == 2:
                obstacles.append(X_pose_hole(X_POSE_HOLE))
            else:
                obstacles.append(Ski_pose_hole(SKI_POSE_HOLE))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            if player.dino_rect.colliderect(obstacle.rect) and not obstacle.checked:
                obstacle.checked = True
                if player.current_pose == obstacle.required_pose:
                    pass
                else:
                    obstacle.speed = -20
                    points = points - 20
                    obstacle.checked = False

        score()

        clock.tick(30)
        pygame.display.update()


def menu():
    global SCREEN, SCREEN_WIDTH, SCREEN_HEIGHT, TEXT_POSITION
    global points
    pygame.init()

    SCREEN_HEIGHT = 600
    SCREEN_WIDTH = 1100
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        text = font.render("Press any Key to Start", True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(STANDING, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 200))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                run = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                SCREEN.fill((255, 255, 255))
                pygame.display.update()
                main()

if __name__ == '__main__':
    menu()