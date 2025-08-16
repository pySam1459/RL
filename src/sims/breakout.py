import cv2
import math
import pygame
import numpy as np
import random
from dataclasses import dataclass, field

from .utils import Vec2
from .base import ISim


__all__ = [
    "BreakoutSim",
    "play_breakout"
]


NROWS = 6
NCOLS = 9

BRICK_W = 8
BRICK_H = 3
BRICK_YOFF = BRICK_H * 3

FRAME_WIDTH  = BRICK_W * NCOLS
FRAME_HEIGHT = int(FRAME_WIDTH * 16 / 10)

BAT_Y = FRAME_HEIGHT - BRICK_H * (NROWS - 3) 
BAT_W = 12
BAT_H = 4
BAT_SPEED = BAT_W // 2
ACTION_SPACE = [0, -1, 1]

BALL_SPEED = 3.0


@dataclass
class Brick:
    x:  int
    y:  int
    w:  int
    h:  int
    yi: int # y index


def init_ball_vel() -> Vec2:
    # Initialize ball velocity with magnitude BALL_SPEED at an angle
    # in [10, 55] degrees from the bat's upward normal (toward bricks)
    angle_degrees = random.uniform(10.0, 55.0)
    side = random.choice([-1.0, 1.0])  # -1: left, 1: right
    theta = math.radians(angle_degrees)
    return Vec2(
        BALL_SPEED * math.sin(theta) * side,
       -BALL_SPEED * math.cos(theta)  # negative y to move upward toward bricks
    )


@dataclass
class Ball:
    pos: Vec2
    vel: Vec2 = field(default_factory=init_ball_vel)


class BreakoutSim(ISim):
    def __init__(self):
        self.batx = (FRAME_WIDTH - BAT_W) // 2
        self.bricks = [Brick(j*BRICK_W, BRICK_YOFF+i*BRICK_H, BRICK_W, BRICK_H, i)
                        for i in range(NROWS) for j in range(NCOLS)]
        
        # ball starts above the middle of the bat
        self.ball = Ball(Vec2(self.batx + BAT_W//2, BAT_Y-1))
        self.score = 0
    
    def tick(self, action: int) -> bool: # returns termination
        # agent makes action
        bat_move = ACTION_SPACE[action]

        self.batx += bat_move * BAT_SPEED
        if self.batx < 0: self.batx = 0
        elif self.batx > FRAME_WIDTH-BAT_W: self.batx = FRAME_WIDTH-BAT_W

        ## use substeps to make sim more accurate
        substeps = 64
        dt = 1.0/substeps
        for _ in range(substeps):
            self.ball.pos += self.ball.vel * dt
            if self.__ball_checks():
                return True
        
        return False ## did not terminate this round
    
    def __ball_checks(self) -> bool:
        if self.__ball_bounds_check():
            return True
        
        for i, br in enumerate(self.bricks):
            if br.x <= self.ball.pos.x <= br.x+br.w and br.y <= self.ball.pos.y <= br.y+br.h:
                # brick bounce
                self.__brick_bounce(br)
                self.score += NROWS - br.yi # top bricks give more score
                self.bricks.pop(i)
                break
        
        if len(self.bricks) == 0:
            return True ## end game if no bricks left
        
        # bat check
        if self.batx <= self.ball.pos.x <= self.batx + BAT_W and BAT_Y <= self.ball.pos.y <= BAT_Y+BAT_H:
            self.__bat_bounce()

        return False

    def __ball_bounds_check(self) -> bool:
        if self.ball.pos.x < 0:
            self.ball.pos.x = -self.ball.pos.x
            self.ball.vel.x *= -1
        elif self.ball.pos.x > FRAME_WIDTH:
            self.ball.pos.x = 2*FRAME_WIDTH - self.ball.pos.x
            self.ball.vel.x *= -1

        if self.ball.pos.y < 0:
            self.ball.pos.y = -self.ball.pos.y
            self.ball.vel.y *= -1

        return self.ball.pos.y > FRAME_HEIGHT ## end game if ball hits bottom

    def __brick_bounce(self, br: Brick) -> None:
        # Determine which side of the brick was hit based on proximity
        dx_left   = abs(self.ball.pos.x - br.x)
        dx_right  = abs((br.x + br.w) - self.ball.pos.x)
        dy_top    = abs(self.ball.pos.y - br.y)
        dy_bottom = abs((br.y + br.h) - self.ball.pos.y)

        min_dx = min(dx_left, dx_right)
        min_dy = min(dy_top, dy_bottom)

        if min_dx < min_dy:
            # Hit vertical side: reflect horizontally and snap outside
            if dx_left < dx_right:
                self.ball.pos.x = br.x - 1e-6
            else:
                self.ball.pos.x = br.x + br.w + 1e-6
            self.ball.vel.x *= -1
        else:
            # Hit horizontal side: reflect vertically and snap outside
            if dy_top < dy_bottom:
                self.ball.pos.y = br.y - 1e-6
            else:
                self.ball.pos.y = br.y + br.h + 1e-6
            self.ball.vel.y *= -1
    
    def __bat_bounce(self) -> None:
        # place ball just above the bat to prevent tunneling
        self.ball.pos.y = BAT_Y - 1
        self.ball.vel.y = -abs(self.ball.vel.y)
        # add small horizontal randomness
        jitter = random.uniform(-0.15, 0.15) * BALL_SPEED
        self.ball.vel.x += jitter
        # normalize to BALL_SPEED
        speed = self.ball.vel.norm()
        if speed == 0:
            self.ball.vel = init_ball_vel()
        else:
            scale = BALL_SPEED / speed
            self.ball.vel *= scale

    def to_image(self) -> np.ndarray:
        # RGB image
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # brick colors vary by row
        for br in self.bricks:
            x0 = max(0, br.x)
            x1 = min(FRAME_WIDTH, br.x + br.w)
            y0 = max(0, br.y)
            y1 = min(FRAME_HEIGHT, br.y + br.h)
            # simple gradient color by brick row index
            r = 255
            g = int(50 + (205 * (NROWS - 1 - br.yi) / max(1, NROWS - 1)))
            b = int(50 + (100 * (br.yi) / max(1, NROWS - 1)))
            img[y0:y1, x0:x1] = (r, g, b)

        # draw bat (cyan)
        bat_x0 = max(0, self.batx)
        bat_x1 = min(FRAME_WIDTH, self.batx + BAT_W)
        bat_y0 = max(0, BAT_Y)
        bat_y1 = min(FRAME_HEIGHT, BAT_Y + BAT_H)
        img[bat_y0:bat_y1, bat_x0:bat_x1] = (0, 255, 255)

        # draw ball (white)
        bx = int(round(self.ball.pos.x))
        by = int(round(self.ball.pos.y))
        if 0 <= bx < FRAME_WIDTH and 0 <= by < FRAME_HEIGHT:
            img[by, bx] = (255, 255, 255)

        return img
    
    def get_reward(self) -> float:
        return self.score - 2  # -2 s.t. reward is negative at start


    def render(self) -> np.ndarray:
        img_rgb = self.to_image()
        scale = 4
        img_rgb_up = cv2.resize(img_rgb, (FRAME_WIDTH * scale, FRAME_HEIGHT * scale), interpolation=cv2.INTER_NEAREST)
        return img_rgb_up



def _ndarray_to_surface(frame_rgb: np.ndarray) -> pygame.Surface:
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.data, (w, h), "RGB")


def play_breakout():
    """Function to play breakout if you are a human!"""
    pygame.init()

    sim = BreakoutSim()
    first = sim.render()
    surf = _ndarray_to_surface(first)
    w, h = surf.get_size()

    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Breakout")

    clock = pygame.time.Clock()
    target_fps = 30

    # key state (held) -> continuous, reliable movement
    holding_left = False
    holding_right = False

    # map pygame keys
    LEFT_KEYS  = {pygame.K_a, pygame.K_LEFT}
    RIGHT_KEYS = {pygame.K_d, pygame.K_RIGHT}
    QUIT_KEYS  = {pygame.K_q, pygame.K_ESCAPE}

    running = True
    try:
        while running:
            # ---- input ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in QUIT_KEYS:
                        running = False
                    if event.key in LEFT_KEYS:
                        holding_left = True
                    if event.key in RIGHT_KEYS:
                        holding_right = True
                elif event.type == pygame.KEYUP:
                    if event.key in LEFT_KEYS:
                        holding_left = False
                    if event.key in RIGHT_KEYS:
                        holding_right = False

            # decide action from held state
            if holding_left and not holding_right:
                action = 1
            elif holding_right and not holding_left:
                action = 2
            else:
                action = 0

            # ---- step & render ----
            terminated = sim.tick(action)
            if terminated:
                sim = BreakoutSim()
                holding_left = holding_right = False

            frame = sim.render()  # numpy BGR frame
            surf = _ndarray_to_surface(frame)

            # blit to window (center if window resized; here we keep fixed size)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # ---- pacing ----
            clock.tick(target_fps)

    finally:
        pygame.quit()
