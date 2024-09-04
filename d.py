cwd = '/home/tuannm84/Desktop/myclip/vtcc-myclip-recommender-system-v2/myclip_recommender_v2/asset/'
class DDPG(object):
    """ Deep Deterministic Policy Gradient Algorithm """

    def __init__(self, env, users_num, items_num, num_actions, STATE_SIZE, output_dim):
        self.env = env
        self.num_states = STATE_SIZE
        self.num_actions = num_actions ## Number of video to be choosed

        # Initialize Actor and Critic networks
        self.critic_net = Critic(self.num_states, self.num_actions)
        self.actor_net = Actor(self.num_states, self.num_actions)

        # Initialize Replay Memory
        self.replay_memory = deque()

        # Initialize time step
        self.time_step = 0
        self.counter = 0

        action_max = [num_actions]
        action_min = [1]
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)
        
        self.embedding_dim = 5
        self.embedding_network = UserVideoEmbedding(users_num, items_num, self.embedding_dim)
        embedding_save_file_dir = os.path.join(cwd, 'dataset/save_weights/user_movie_embedding_case4.h5') #m_g_model_weights.weights.h5'
        assert os.path.exists(embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."
        self.embedding_network.built = True
        self.embedding_network.load_weights(embedding_save_file_dir, by_name = True, skip_mismatch = True)
        
        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim, output_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1, STATE_SIZE, 100))])
        
    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)

    def add_experience(self, observation_1, observation_2, action, reward, done):
        self.observation_1 = observation_1 # previous 
        self.observation_2 = observation_2 # newest 
        self.action = action
        self.reward = reward
        self.done = done
        self.replay_memory.append((self.observation_1, self.observation_2, self.action, self.reward, self.done))
        self.time_step += 1
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

    def minibatches(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        # state t
        self.state_t_batch = np.array([item[0] for item in batch])
        # state t+1
        self.state_t_1_batch = np.array([item[1] for item in batch])
        self.action_batch = np.array([item[2] for item in batch]).reshape(len(batch), self.num_actions)
        self.reward_batch = np.array([item[3] for item in batch])
        self.done_batch = np.array([item[4] for item in batch])

    def train(self):
        # Sample a random minibatch of N transitions from replay memory
        self.minibatches()
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.state_t_1_batch)
        # Q'(s_i+1,a_i+1)
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_batch, self.action_t_1_batch)
        self.y_i_batch = []

        for i in range(BATCH_SIZE):
            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            else:
                self.y_i_batch.append(self.reward_batch[i] + GAMMA * q_t_1[i][0])

        self.y_i_batch = np.array(self.y_i_batch).reshape(len(self.y_i_batch), 1)

        # Update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_batch, self.action_batch, self.y_i_batch)

        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self.state_t_batch)

        if is_grad_inverter:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch, action_for_delQ)
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a, action_for_delQ)
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch, action_for_delQ)[0]

        # Train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(self.state_t_batch, self.del_Q_a)

        # Update target Critic and Actor networks
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
                     
                     
    def recommend_item(self, action, all_items, old_watched, top_k=False, items_ids=None):
        if items_ids is None:
            items_ids = np.array(list(set(all_items) - set(old_watched)))
            # items_ids = np.array(list(set(self.items_list) - set(recommended_items)))

        
        items_ebs = self.embedding_network.get_layer('video_embedding')(items_ids)
        action = tf.expand_dims(action, axis=1)
        action = tf.transpose(action)
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.reduce_sum((items_ebs* action), axis=1, keepdims=True), perm=(1, 0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.transpose(tf.reduce_sum((items_ebs* action), axis=1, keepdims=True), perm=(1, 0)))
            return items_ids[item_idx]
