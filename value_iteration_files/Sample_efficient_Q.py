def run_episodes(softmax_enabled=False, experience_replay=False,epsilon=0.1, temperature = 0.5):
    state = 12
    ship_taken = False
    cumulative_reward = 0
    penalty = 0
    for i in range(10000):
        while True:
            if not softmax_enabled:
                if random.uniform(0, 1) < epsilon:
                    # Check the action space
                    action = np.random.choice(4)
                else:
                    action = np.argmax(qtable[state])
            else:
                action = softmax(state,temperature)
            rob.direction = action_dict[action]
            if not should_slip(rob,env):
                move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            #print(new_state)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
            cumulative_reward += reward
            if reward == -10:
                penalty += 1
            if experience_replay:
                memory.append([state, action, reward, new_state])
            else:
                max = np.max(qtable[new_state])
                value = qtable[state, action]
                newval = value + alpha * (reward + (gamma * max) - value)
                qtable[state][action] = newval

            state = new_state
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                # if divmod(i, 10)[1] == 0 and i<100:
                    # print("total reward:", cumulative_reward)
                if experience_replay:
                    replay()
                rewards.append(cumulative_reward)
                # print(np.argmax(qtable[10]))
                if random.uniform(0,100) <101: # chance of random start (100% def)
                    state = 15
                    while not env.is_on_ice(rob):
                        rob.x = random.randint(0,3)
                        #print('rob x: {0}'.format(rob.x))
                        rob.y = random.randint(0,3)
                    state = 4 * rob.y + rob.x
                    #print("placed")
                else:
                    state = 12
                    rob.x = STARTING_POS[0]
                    rob.y = STARTING_POS[1]

                rob.direction = ""
                break

    return cumulative_reward, penalty
