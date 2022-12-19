#include "camera.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>

int main() {
    camera c = camera();

    mandelbox box = mandelbox();

    sf::Uint8* pixels;
    cudaMallocManaged(&pixels, (int) (c.WIDTH * c.HEIGHT * 4) * sizeof(sf::Uint8));

    sf::RenderWindow window(sf::VideoMode((int) c.WIDTH, (int) c.HEIGHT), "Ray Marcher");

    sf::Image image;
    sf::Texture texture;
    sf::Sprite sprite;

    double dir = 1.0;

    sf::Clock clock;
    float lastTime = 0;

    while (window.isOpen()) {
        c.render(pixels, box);

        image.create((int) c.WIDTH, (int) c.HEIGHT, pixels);
        texture.loadFromImage(image);
        sprite.setTexture(texture);

        window.draw(sprite);
        window.display();

        sf::Event event{};
        while (window.pollEvent(event)){
            if (event.type == sf::Event::Closed)
                window.close();
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)){
            c.posX -= 0.04;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)){
            c.posX += 0.04;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)){
            c.posZ -= 0.04;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)){
            c.posZ += 0.6;
        }

        //box.renderScale += 0.0005 * dir;
        if (box.renderScale > 2 || box.renderScale < 1.6) {
            dir *= -1;
        }

        float currentTime = clock.restart().asSeconds();
        std::cout << 1.f / currentTime << '\n';
        lastTime = currentTime;

    }

    cudaFree(pixels);

    return 0;
}