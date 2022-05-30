from snake import Direction, Fruit, Snake, Wall
import pygame

pygame.init()
pygame.display.init()
window = pygame.display.set_mode((512, 512))
pygame.display.set_caption('Snake')

snake = Snake(16, 512, (0, 0, 255))
walls = Wall(16, 512, (0, 0, 0), snake.body)
fruit = Fruit(16, 512, (255, 0, 0), snake.body, walls.segments)

run = True
while run:
    pygame.time.delay(100)

    for event in pygame.event.get():
        run = not (event.type == pygame.QUIT)

    key = pygame.key.get_pressed()
    if key[pygame.K_LEFT]:
        snake.change_direction(Direction.LEFT)
    elif key[pygame.K_DOWN]:
        snake.change_direction(Direction.DOWN)
    elif key[pygame.K_RIGHT]:
        snake.change_direction(Direction.RIGHT)
    elif key[pygame.K_UP]:
        snake.change_direction(Direction.UP)

    snake.move()
    snake.eat_check(fruit, walls.segments)
    if snake.is_dead(walls.segments):
        font = pygame.font.SysFont('ariel', 60, True)
        text = font.render('Game Over', True, (0, 255, 255))
        text_rect = text.get_rect(center=(256, 256))
        window.blit(text, text_rect)

        pygame.display.update()
        pygame.time.delay(2000)

        snake.reset()
        walls.reset(snake.body)
        fruit.reset(snake.body, walls.segments)

    window.fill((255, 255, 255))
    snake.render(window)
    walls.render(window)
    fruit.render(window)
    pygame.display.flip()
