import numpy as np


def PSO(obj, Boundaries, num_particles, max_iter, w, c1, c2, maximizing=False):
    num_dimensions = len(Boundaries)

    particles = np.random.uniform(Boundaries[:, 0], Boundaries[:, 1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = particles.copy()
    personal_best_values = np.asarray([obj(_) for _ in particles])
    global_best_position = particles[np.argmax(personal_best_values)] if maximizing \
        else particles[np.argmin(personal_best_values)]
    global_best_value = np.max(personal_best_values) if maximizing else np.min(personal_best_values)
    history = list()

    for iteration in range(max_iter):
        for i in range(num_particles):
            particle = particles[i]

            # Evaluate objective function
            particle_value = obj(particle)

            if maximizing:
                if particle_value > personal_best_values[i]:
                    personal_best_values[i] = particle_value
                    personal_best_positions[i] = particle.copy()

                if particle_value > global_best_value:
                    global_best_value = particle_value
                    global_best_position = particle.copy()
            else:
                if particle_value < personal_best_values[i]:
                    personal_best_values[i] = particle_value
                    personal_best_positions[i] = particle.copy()

                # Update global best
                if particle_value < global_best_value:
                    global_best_value = particle_value
                    global_best_position = particle.copy()

            # Update velocity
            velocity = velocities[i]
            inertia_term = w * velocity
            cognitive_term = c1 * np.random.rand() * (personal_best_positions[i] - particle)
            social_term = c2 * np.random.rand() * (global_best_position - particle)
            velocities[i] = inertia_term + cognitive_term + social_term

            # Update position
            particles[i] += velocities[i]

            # Check position bounds
            particles[i] = np.clip(particles[i], Boundaries[:, 0], Boundaries[:, 1])
        history.append(global_best_value)

    return [global_best_position, global_best_value, history]
