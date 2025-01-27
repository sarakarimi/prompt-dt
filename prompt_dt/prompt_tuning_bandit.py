
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prompt_dt.prompt_evaluate_episodes import prompt_evaluate_episode_rtg
from prompt_dt.prompt_utils import discount_cumsum


class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        h_dim = 16
        self.network = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            # nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            # nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


# NeuralUCB-like algorithm with the TrainNN method
class NeuralBandit:
    def __init__(self, input_dim, n_segments, segment_length, regularization_lambda=1.0, learning_rate=0.01,
                 steps_per_update=100):
        self.n_segments = n_segments
        self.segment_length = segment_length

        self.segment_reward_models = []
        self.initial_weights = []
        self.optimizers = []
        self.input_dim = input_dim
        for _ in range(n_segments):
            reward_model = RewardModel(input_dim)

            # save initial state dict instead
            initial_weights = {k: v.clone().detach() for k, v in reward_model.state_dict().items()}
            optimizer = optim.SGD(reward_model.parameters(), lr=learning_rate, weight_decay=regularization_lambda)

            self.initial_weights.append(initial_weights)
            self.optimizers.append(optimizer)
            self.segment_reward_models.append(
                reward_model)  # could also try this with shared backbone and separate heads...

        self.regularization_lambda = regularization_lambda
        self.learning_rate = learning_rate
        self.steps_per_update = steps_per_update

        self.loss_fn = nn.MSELoss()
        self.segments = [[] for _ in range(n_segments)]  # we have a list for each segment...
        self.rewards = []  # Stores all observed rewards

    def store_observation(self, segments, reward):
        """Store observed context and reward."""
        assert len(segments) == self.n_segments
        for segment_idx, segment in enumerate(segments):
            self.segments[segment_idx].append(segment)
        self.rewards.append(reward)

    def update_model(self):
        """Train the neural network using the observed data."""
        if len(self.rewards) < 2:
            return  # No data to train on yet

        all_segment_model_losses = [[] for _ in range(self.n_segments)]
        for segment_idx in range(self.n_segments):

            # randomly shuffle the rewards and segments in the same way
            shuffled_indices = np.random.permutation(len(self.rewards))
            self.rewards = [self.rewards[i] for i in shuffled_indices]
            for segment_shuffle_idx in range(self.n_segments):
                self.segments[segment_shuffle_idx] = [self.segments[segment_shuffle_idx][i] for i in shuffled_indices]

            # Convert data to tensors
            contexts_tensor = torch.from_numpy(np.array(self.segments[segment_idx])).float()
            rewards_tensor = torch.from_numpy(np.array(self.rewards)).float()

            # Perform multiple steps of gradient descent
            losses = []
            for step in range(self.steps_per_update):
                self.optimizers[segment_idx].zero_grad()

                # Compute predictions and loss
                predictions = self.segment_reward_models[segment_idx](contexts_tensor)
                loss = self.loss_fn(predictions.squeeze(), rewards_tensor.squeeze())
                losses.append(loss.item())

                # Add regularization term
                # reg_term = self.regularization_lambda * sum(p.norm(2) ** 2 for p in self.segment_reward_models[segment_idx].parameters())
                # total_loss = loss + reg_term
                total_loss = loss

                # Backpropagation and optimization
                total_loss.backward()

                # clip grad norm, this seems essential due to outliers!
                torch.nn.utils.clip_grad_norm_(self.segment_reward_models[segment_idx].parameters(), 10.0)

                self.optimizers[segment_idx].step()
            all_segment_model_losses[segment_idx].extend(losses)

        return all_segment_model_losses

    def predict(self, context):
        """Predict the reward for a given context."""
        segment_preds = []
        context_tensor = torch.from_numpy(np.array(context)).to(torch.float32)
        with torch.no_grad():
            for segment_idx in range(self.n_segments):
                # reward_pred = self.segment_reward_models[segment_idx](context_tensor).item()
                reward_pred = self.segment_reward_models[segment_idx](context_tensor).squeeze()
                segment_preds.append(reward_pred)

        return segment_preds


def all_possible_segments(model, expert_prompt_trajs, rtg_scale, state_dim, act_dim, traj_prompt_j, traj_prompt_h,
                          device,
                          context_len, eval_batch_size, bandit_use_transformer_features=False, bandit_feature_dim=None):
    # create alist of all possible segments
    segments_raw = []  # all possible traj prompt segments of length h * 3, no encoding
    segments_features = []  # all possible traj prompt segments of length h * 3, using some feature encoding
    segment_idx = []
    traj_idxs = []
    for traj_prompt_segment_idx in range(len(expert_prompt_trajs)):
        prompt_traj = expert_prompt_trajs[traj_prompt_segment_idx]
        return_to_go = discount_cumsum(prompt_traj["rewards"], gamma=1.)

        for i in range(0, len(prompt_traj["observations"]) - traj_prompt_h + 1):
            segment_observations = prompt_traj["observations"][i:i + traj_prompt_h]
            segment_actions = prompt_traj["actions"][i:i + traj_prompt_h]
            segment_rtgs = return_to_go[i:i + traj_prompt_h]  # prompt_traj["rewards"][i:i + traj_prompt_h]
            segment_idx.append((i, i + traj_prompt_h))
            segment_timesteps = np.arange(i, i + traj_prompt_h, step=1)
            segment_masks = np.ones((traj_prompt_h))

            traj_idxs.append(traj_prompt_segment_idx)

            segment_rtgs = segment_rtgs / rtg_scale

            # stack and interleave observations, actions, and rtgs into (r, s, a, r, s, a ...)
            arm_features_raw = []
            for t in range(traj_prompt_h):
                if state_dim > 1:
                    for obs_scalar in segment_observations[t]:
                        arm_features_raw.append(obs_scalar)
                else:
                    arm_features_raw.append(segment_observations[t])

                if act_dim > 1:
                    for act_scalar in segment_actions[t]:
                        arm_features_raw.append(act_scalar)
                else:
                    arm_features_raw.append(segment_actions[t])
                arm_features_raw.append(segment_rtgs[t][0])
            segments_raw.append(arm_features_raw)

            if bandit_use_transformer_features:
                # get the transformer features for the current segment
                segment_timestep_tensor = torch.from_numpy(segment_timesteps).to(torch.int64).to(device).reshape(1,
                                                                                                                 traj_prompt_h)
                segment_masks_tensor = torch.from_numpy(segment_masks).to(torch.int64).to(device).reshape(1,
                                                                                                          traj_prompt_h)

                segment_states_tensor = torch.from_numpy(segment_observations).to(torch.float32).to(device).reshape(1,
                                                                                                                    traj_prompt_h,
                                                                                                                    state_dim)
                segment_actions_tensor = torch.from_numpy(segment_actions).to(torch.float32).to(device).reshape(1,
                                                                                                                traj_prompt_h,
                                                                                                                act_dim)
                segment_rtgs_tensor = torch.from_numpy(segment_rtgs).to(torch.float32).to(device).reshape(1,
                                                                                                          traj_prompt_h,
                                                                                                          1)

                # repeat for each segment we ought to have, to get right shape of input tensors, we just look at the first one, though...
                # TODO, double check this is correct
                segment_timestep_tensor = segment_timestep_tensor.repeat(1, traj_prompt_j)
                segment_masks_tensor = segment_masks_tensor.repeat(1, traj_prompt_j)
                segment_states_tensor = segment_states_tensor.repeat(1, traj_prompt_j, 1)
                segment_actions_tensor = segment_actions_tensor.repeat(1, traj_prompt_j, 1)
                segment_rtgs_tensor = segment_rtgs_tensor.repeat(1, traj_prompt_j, 1)

                prompt = (
                    segment_states_tensor, segment_actions_tensor, None, None, segment_rtgs_tensor,
                    segment_timestep_tensor,
                    segment_masks_tensor)

                _, act_preds, _, rtg_features, state_features, action_features = model.forward(
                    torch.zeros((eval_batch_size, context_len, state_dim), dtype=torch.float32, device=device),
                    torch.zeros((eval_batch_size, context_len, act_dim), dtype=torch.float32, device=device),
                    None,
                    torch.zeros((eval_batch_size, context_len, 1), dtype=torch.float32, device=device),
                    torch.arange(start=0, end=context_len, step=1).repeat(eval_batch_size, 1).to(device),
                    attention_mask=torch.ones((context_len)).repeat(eval_batch_size, 1).to(device),
                    prompt=prompt, features=True,
                )

                assert action_features.shape == (
                    1, traj_prompt_j * traj_prompt_h + context_len, bandit_feature_dim)
                segment_features_transformer = action_features[0, traj_prompt_h - 1, :].cpu().detach().numpy()
                segments_features.append(segment_features_transformer)

    return segments_raw, segments_features, segment_idx, traj_idxs


def select_segments(mab, eval_batch_size, expert_prompt_trajs, traj_idxs, segments_raw, segments_features, segment_idx,
                    epsilon,
                    epsilon_decay, state_dim, act_dim, device, traj_prompt_j, traj_prompt_h,
                    bandit_use_transformer_features, info, variant):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']

    if np.random.rand() < epsilon:
        selected_segment_idxs = [np.random.randint(len(segments_raw)) for _ in range(mab.n_segments)]
        selected_segments_start_end = [segment_idx[arm_idx] for arm_idx in selected_segment_idxs]
        print("MAB: exploring random arm, segment idxs: ", selected_segments_start_end)
    else:
        with torch.no_grad():
            if bandit_use_transformer_features:
                rewards_pred = mab.predict(segments_features)
            else:
                rewards_pred = mab.predict(segments_raw)

        rewards_pred = np.array(rewards_pred).T
        selected_segment_idxs = np.argmax(rewards_pred, axis=0)
        assert len(selected_segment_idxs) == mab.n_segments
        selected_segments_start_end = [segment_idx[arm_idx] for arm_idx in selected_segment_idxs]
        print(f"MAB: exploiting segments {selected_segment_idxs}, start end: {selected_segments_start_end}")

    if epsilon >= 0.0:
        epsilon -= epsilon_decay


    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

    for seg_counter, seg_idx in enumerate(selected_segment_idxs):
        prompt_traj = expert_prompt_trajs[traj_idxs[seg_idx]]
        start, end = selected_segments_start_end[seg_counter]

        # get sequences from dataset
        s.append(prompt_traj['observations'][start:end].reshape(1, -1, state_dim))
        a.append(prompt_traj['actions'][start:end].reshape(1, -1, act_dim))
        r.append(prompt_traj['rewards'][start:end].reshape(1, -1, 1))
        if 'terminals' in prompt_traj:
            d.append(prompt_traj['terminals'][start:end].reshape(1, -1))
        else:
            d.append(prompt_traj['dones'][start:end].reshape(1, -1))
        timesteps.append(np.arange(start, start + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
        rtg.append(discount_cumsum(prompt_traj['rewards'][start:], gamma=1.)[:s[-1].shape[1]].reshape(1, -1, 1))
        # if rtg[-1].shape[1] <= s[-1].shape[1]:
        #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        max_len = end - start
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        if not variant['no_state_normalize']:
            s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    n_traj_prompt_segments = traj_prompt_j
    traj_prompt_seg_len = traj_prompt_h
    s = s.reshape(eval_batch_size, n_traj_prompt_segments * traj_prompt_seg_len, state_dim)
    a = a.reshape(eval_batch_size, n_traj_prompt_segments * traj_prompt_seg_len, act_dim)
    rtg = rtg.reshape(eval_batch_size, n_traj_prompt_segments * traj_prompt_seg_len, 1)
    timesteps = timesteps.reshape(eval_batch_size, n_traj_prompt_segments * traj_prompt_seg_len)
    mask = mask.reshape(eval_batch_size, n_traj_prompt_segments * traj_prompt_seg_len)
    return s, a, rtg, timesteps, mask, selected_segment_idxs, epsilon

def rollout_bandit(env, model, max_test_ep_len, rtg_target, rtg_scale, act_dim, state_dim, device,
                   traj_prompt_timesteps, traj_prompt_states, traj_prompt_actions, traj_prompt_rtgs,
                   traj_prompt_masks, info,
                   variant, traj_prompt_j, traj_prompt_h):
    n_traj_prompt_segments = traj_prompt_j
    traj_prompt_seg_len = traj_prompt_h

    prompt = (
        traj_prompt_states, traj_prompt_actions, None, None, traj_prompt_rtgs, traj_prompt_timesteps, traj_prompt_masks)

    with torch.no_grad():
        ret, _ = prompt_evaluate_episode_rtg(
            env,
            state_dim,
            act_dim,
            model,
            max_ep_len=max_test_ep_len,
            scale=rtg_scale,
            target_return=rtg_target[0] / rtg_scale,
            state_mean=info['state_mean'],
            state_std=info['state_std'],
            device=device,
            prompt=prompt,
            no_r=variant['no_r'],
            no_rtg=variant['no_rtg'],
            no_state_normalize=variant['no_state_normalize']
        )

    prompt_state_segments = []

    for segment in range(n_traj_prompt_segments):
        prompt_state_segments.append(
            traj_prompt_states[
            0,  # index of the only segment we have, because this code is just for prompts with containing one segment
            segment * traj_prompt_seg_len:(segment + 1) * traj_prompt_seg_len
            ].cpu().numpy())

    return prompt_state_segments, ret


def update_mab(mab, sparse_reward_sum, selected_segment_idxs, segments_raw, segments_features,
               bandit_use_transformer_features):
    # update MAB
    store_segments = []
    for j_idx, segment_update_idx in enumerate(selected_segment_idxs):
        # start, end = selected_segments_start_end[j_idx]

        if bandit_use_transformer_features:
            store_segments.append(segments_features[segment_update_idx])
        else:
            store_segments.append(segments_raw[segment_update_idx])

    mab.store_observation(store_segments, sparse_reward_sum)  # USE SPARSE REWARD HERE!

    # reset reward model every time...
    for model_idx in range(len(mab.segment_reward_models)):
        initial_parameter = mab.initial_weights[model_idx]
        mab.segment_reward_models[model_idx].load_state_dict({k: v.clone() for k, v in initial_parameter.items()})

    for reward_model in mab.segment_reward_models:
        reward_model.train()
    losses = mab.update_model()

    return losses


def prompt_tuning_bandit(model, prompt_trajectory, env, info, variant, env_name, wandb):
    logs = dict()
    seed = variant['seed']
    print("Running bandit on seed", seed)
    # seeding
    random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # params
    eval_batch_size = 1
    num_rollouts = 250
    epsilon = 1.0
    # decay epsilon over the first 20% of num_rollouts to zero
    epsilon_decay = 1.0 / (num_rollouts * 0.2)
    rtg_scale = info["scale"]
    rtg_target = info["env_targets"]
    max_test_ep_len = info["max_ep_len"]
    act_dim = info["act_dim"]
    state_dim = info["state_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_len = variant["K"]
    traj_prompt_j = variant["num_traj_prompt_j"]
    traj_prompt_h = variant['prompt_length']
    bandit_use_transformer_features = variant['bandit_use_transformer_features']

    if bandit_use_transformer_features:
        bandit_feature_dim = variant['embed_dim']
    else:
        n_segment_tokens = traj_prompt_h * state_dim
        n_segment_tokens += traj_prompt_h * act_dim
        n_segment_tokens += traj_prompt_h * 1
        bandit_feature_dim = n_segment_tokens

    rollouts_rewards = []
    rollout_prompt_states = []

    mab = NeuralBandit(
        input_dim=bandit_feature_dim,
        n_segments=traj_prompt_j,
        segment_length=traj_prompt_h * (state_dim + act_dim + 1),
        regularization_lambda=0.0,
        learning_rate=0.01,
        steps_per_update=1000
    )

    # get all possible segments
    segments_raw, segments_features, segment_idx, traj_idxs = all_possible_segments(model, prompt_trajectory, rtg_scale,
                                                                                    state_dim, act_dim,
                                                                                    traj_prompt_j, traj_prompt_h,
                                                                                    device, context_len,
                                                                                    eval_batch_size,
                                                                                    bandit_use_transformer_features,
                                                                                    bandit_feature_dim)
    # bandit select segments
    for rollout_idx in range(num_rollouts):
        traj_prompt_states, traj_prompt_actions, traj_prompt_rtgs, traj_prompt_timesteps, traj_prompt_masks, selected_segment_idxs, epsilon = select_segments(
            mab, eval_batch_size, prompt_trajectory, traj_idxs, segments_raw, segments_features, segment_idx, epsilon,
            epsilon_decay, state_dim, act_dim, device, traj_prompt_j, traj_prompt_h, bandit_use_transformer_features, info, variant)

        # rollout bandit selected prompts and select the best arm
        prompt_state_segments, ret = rollout_bandit(env, model, max_test_ep_len, rtg_target,
                                                    rtg_scale,
                                                    act_dim, state_dim, device, traj_prompt_timesteps,
                                                    traj_prompt_states,
                                                    traj_prompt_actions, traj_prompt_rtgs,
                                                    traj_prompt_masks, info, variant, traj_prompt_j, traj_prompt_h)

        rollouts_rewards.append(ret)
        rollout_prompt_states.append(prompt_state_segments)
        print("mab results", ret)

        # update MAB
        train_losses = update_mab(mab, ret, selected_segment_idxs, segments_raw, segments_features, bandit_use_transformer_features)

        if train_losses is not None:
            logs[f'training/{env_name}_train_loss_mean'] = np.mean(train_losses)
            logs[f'training/{env_name}_train_loss_std'] = np.std(train_losses)

        logs.update({f'{env_name}_target_0_return_mean': ret})
        wandb.log(logs)

    return rollouts_rewards, logs
