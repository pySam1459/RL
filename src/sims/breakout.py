import math
import random
import cv2
import numpy as np
from dataclasses import dataclass

from base import ISim


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


def init_ball_vel() -> list[float]:
    # Initialize ball velocity with magnitude BALL_SPEED at an angle
    # in [10, 55] degrees from the bat's upward normal (toward bricks)
    angle_degrees = random.uniform(10.0, 55.0)
    side = random.choice([-1.0, 1.0])  # -1: left, 1: right
    theta = math.radians(angle_degrees)
    return [
        BALL_SPEED * math.sin(theta) * side,
       -BALL_SPEED * math.cos(theta)  # negative y to move upward toward bricks
    ]





@dataclass
class Brick:
    x:  int
    y:  int
    w:  int
    h:  int
    yi: int # y index


class BreakoutSim(ISim):
    def __init__(self):
        self.batx = (FRAME_WIDTH - BAT_W) // 2
        self.bricks = [Brick(j*BRICK_W, BRICK_YOFF+i*BRICK_H, BRICK_W, BRICK_H, i)
                        for i in range(NROWS) for j in range(NCOLS)]
        
        # ball starts above the middle of the bat
        self.ball_pos = [self.batx + BAT_W//2, BAT_Y-1]
        self.ball_vel = init_ball_vel()

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
            self.ball_pos[0] += self.ball_vel[0] * dt
            self.ball_pos[1] += self.ball_vel[1] * dt
            if self.__ball_checks():
                return True
        
        return False ## did not terminate this round
    
    def __ball_checks(self) -> bool:
        if self.__ball_bounds_check():
            return True
        
        for i, br in enumerate(self.bricks):
            if br.x <= self.ball_pos[0] <= br.x+br.w and br.y <= self.ball_pos[1] <= br.y+br.h:
                # brick bounce
                self.__brick_bounce(br)
                self.score += NROWS - br.yi # top bricks give more score
                self.bricks.pop(i)
                break
        
        if len(self.bricks) == 0:
            return True ## end game if no bricks left
        
        # bat check
        if self.batx <= self.ball_pos[0] <= self.batx + BAT_W and BAT_Y <= self.ball_pos[1] <= BAT_Y+BAT_H:
            self.__bat_bounce()

        return False

    def __ball_bounds_check(self) -> bool:
        if self.ball_pos[0] < 0:
            self.ball_pos[0] = -self.ball_pos[0]
            self.ball_vel[0] *= -1
        elif self.ball_pos[0] > FRAME_WIDTH:
            self.ball_pos[0] = 2*FRAME_WIDTH - self.ball_pos[0]
            self.ball_vel[0] *= -1

        if self.ball_pos[1] < 0:
            self.ball_pos[1] = -self.ball_pos[1]
            self.ball_vel[1] *= -1

        return self.ball_pos[1] > FRAME_HEIGHT ## end game if ball hits bottom

    def __brick_bounce(self, br: Brick) -> None:
        # Determine which side of the brick was hit based on proximity
        dx_left   = abs(self.ball_pos[0] - br.x)
        dx_right  = abs((br.x + br.w) - self.ball_pos[0])
        dy_top    = abs(self.ball_pos[1] - br.y)
        dy_bottom = abs((br.y + br.h) - self.ball_pos[1])

        min_dx = min(dx_left, dx_right)
        min_dy = min(dy_top, dy_bottom)

        if min_dx < min_dy:
            # Hit vertical side: reflect horizontally and snap outside
            if dx_left < dx_right:
                self.ball_pos[0] = br.x - 1e-6
            else:
                self.ball_pos[0] = br.x + br.w + 1e-6
            self.ball_vel[0] *= -1
        else:
            # Hit horizontal side: reflect vertically and snap outside
            if dy_top < dy_bottom:
                self.ball_pos[1] = br.y - 1e-6
            else:
                self.ball_pos[1] = br.y + br.h + 1e-6
            self.ball_vel[1] *= -1
    
    def __bat_bounce(self) -> None:
        # place ball just above the bat to prevent tunneling
        self.ball_pos[1] = BAT_Y - 1
        self.ball_vel[1] = -abs(self.ball_vel[1])
        # add small horizontal randomness
        jitter = random.uniform(-0.15, 0.15) * BALL_SPEED
        self.ball_vel[0] += jitter
        # normalize to BALL_SPEED
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed == 0:
            self.ball_vel = init_ball_vel()
        else:
            scale = BALL_SPEED / speed
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

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
        bx = int(round(self.ball_pos[0]))
        by = int(round(self.ball_pos[1]))
        if 0 <= bx < FRAME_WIDTH and 0 <= by < FRAME_HEIGHT:
            img[by, bx] = (255, 255, 255)

        return img

    def render(self) -> None:
        img_rgb = self.to_image()
        scale = 4
        img_rgb_up = cv2.resize(img_rgb, (FRAME_WIDTH * scale, FRAME_HEIGHT * scale), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_rgb_up, cv2.COLOR_RGB2BGR)
        cv2.imshow("Breakout", img_bgr)
        cv2.waitKey(1)

