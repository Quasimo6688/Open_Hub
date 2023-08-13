import pygame
import random

# 初始化pygame
pygame.init()
game_over = False


# 设置屏幕尺寸为手机竖屏尺寸
WIDTH, HEIGHT = 720, 1280

# 设置颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 87, 51), (51, 255, 87), (51, 87, 255), (255, 51, 255), (255, 255, 51)]

# 创建屏幕和时钟对象
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker Game")
clock = pygame.time.Clock()

# 定义小球类
class Ball:
    def __init__(self):
        self.radius = 8
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 8
        self.dx = random.choice([-1, 1])
        self.dy = random.choice([-1, 1])

    def move(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

    def draw(self):
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.radius)

    def bounce(self, axis):
        if axis == 'x':
            self.dx *= -1
        elif axis == 'y':
            self.dy *= -1
# 定义挡板类
class Paddle:
    def __init__(self):
        self.width = 100
        self.height = 10
        self.x = (WIDTH - self.width) // 2
        self.y = HEIGHT - 40
    
        self.speed = 9
        self.dx = 0

    def move(self):
        self.x += self.dx * self.speed
        if self.x < 0:
            self.x = 0
        elif self.x > WIDTH - self.width:
            self.x = WIDTH - self.width

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))

# 定义砖块类
class Brick:
    def __init__(self, x, y, color, hits):
        self.x = x
        self.y = y
        self.width = 80
        self.height = 40
        self.color = color
        self.hits = hits

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

# 创建对象
ball = Ball()
balls = [ball]
paddle = Paddle()
bricks = []

# 初始化砖块
brick_rows = 10
brick_cols = 8
brick_width = 80
brick_height = 40
brick_gap = 10
brick_start_x = (WIDTH - (brick_width + brick_gap) * brick_cols) // 2
brick_start_y = 200

for row in range(brick_rows):
    for col in range(brick_cols):
        brick_x = brick_start_x + (brick_width + brick_gap) * col
        brick_y = brick_start_y + (brick_height + brick_gap) * row
        hits = 2 + (row * 2)
        brick = Brick(brick_x, brick_y, COLORS[row % len(COLORS)], hits)
        bricks.append(brick)
# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                paddle.dx = -1
            elif event.key == pygame.K_RIGHT:
                paddle.dx = 1
            elif event.key == pygame.K_SPACE:  # 按下空格键暂停/继续游戏
                for ball in balls:
                    ball.dx *= 0
                    ball.dy *= 0
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                paddle.dx = 0

    # 移动球和挡板
    for ball in balls:
        ball.move()
    paddle.move()

    # 碰撞检测
    # 检查球是否与屏幕边缘碰撞
    for ball in balls:
        if ball.x - ball.radius <= 0 or ball.x + ball.radius >= WIDTH:
            ball.bounce('x')
        if ball.y - ball.radius <= 0:
            ball.bounce('y')
        if ball.y + ball.radius >= HEIGHT:
            balls.remove(ball)

        # 检查球是否与挡板碰撞
        if (ball.y + ball.radius >= paddle.y) and (ball.x >= paddle.x and ball.x <= paddle.x + paddle.width):
            ball.bounce('y')

        # 检查球是否与砖块碰撞
        for brick in bricks:
            if (ball.x + ball.radius > brick.x and ball.x - ball.radius < brick.x + brick.width) and \
               (ball.y + ball.radius > brick.y and ball.y - ball.radius < brick.y + brick.height):
                ball.bounce('y')
                brick.hits -= 1
                if brick.hits <= 0:
                    bricks.remove(brick)
                    if random.random() < 0.15:  # 15%的几率生成新的球
                        new_ball = Ball()
                        new_ball.x = brick.x + brick.width // 2
                        new_ball.y = brick.y + brick.height // 2
                        balls.append(new_ball)
    # 检查游戏是否结束
    if len(balls) == 0:
        game_over = True                    

    # 绘制
    screen.fill(WHITE)
    for ball in balls:
        ball.draw()
    paddle.draw()
    for brick in bricks:
        brick.draw()

    # 如果游戏结束，显示中央的动态字幕
    if game_over:
        font = pygame.font.SysFont(None, 74)
        text = font.render('游戏失败', True, BLACK)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
