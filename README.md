🧠 Bio Brain Snake
Current record: 86 apples in one life. Not bad for a brain grown from zero.

🧬 What makes this brain special?
This isn't your standard neural net. There's no gradient descent, no loss function, nobody computing derivatives. Instead, the snake gets a brain full of exotic neuron types, and evolution shapes them over generations.

19 neuron types — each with a personality. You've got the classics: excitatory and inhibitory neurons, the yin and yang of any brain. Then there are the survival instincts — food-seeking neurons that fire when an apple is near, and wall-avoidance neurons that scream when the edge is close. Loop detectors sense when the snake is about to trap itself. Memory neurons hold onto information across steps, while risk and novelty neurons evaluate danger and curiosity, constantly pushing the snake to explore or play it safe.

Things get weirder from there. Hormone sources and sinks release and absorb global chemical signals that wash across the whole brain. Gates open and close pathways depending on hormone levels. Oscillators create rhythmic patterns, like a heartbeat for decision-making. And the crown jewel: Q, K, and V neurons — spiking neuron implementations of transformer-style attention. The snake pays attention, but with biology, not matrix multiplication.

A hormonal system runs in the background. Neurons can release hormones into a shared pool, and gates react to them. A stress hormone might close off exploration circuits when the snake is cornered. A feeding hormone might amplify food-seeking signals after a long hunger. It's messy, decentralized, and surprisingly effective.

The brain organizes itself into modules. Up to 40 of them. During evolution, whole modules get swapped between parents like LEGO blocks, letting good circuits spread through the population without being torn apart.

Evolution adapts its own pace. There's a "Snowball" system: when a snake breaks a record, the mutation rate drops tenfold to protect the breakthrough. When the population stagnates, mutations crank back up to escape the plateau. It's evolution doing meta-learning.

The snake learns within its own lifetime too. Hebbian updates strengthen connections between neurons that fire together. Reward signals reinforce good decisions. Punishment signals weaken bad ones. It's not just evolving between generations — it's learning in real time.

Weak connections don't linger. Any synapse with a strength below 0.01 gets pruned automatically. The brain stays lean.

And the whole thing compresses into a string. The full genome — every neuron, every connection, every parameter — gets packed into a base-26 string with a checksum. You could email a snake brain.

🏆 The result
Field size: 20×20. Record: 86 apples in one life. The snake learned to avoid itself, plan efficient routes, and escape dead ends — all through evolution alone. Nobody programmed those behaviors. They emerged.

🚀 Run it yourself
All you need is Rust 1.70+. No GPU, no CUDA, no cloud. Just your machine and a bit of curiosity.

bash:
git clone https://github.com/hicorn/bio-brain-snake.git
cd bio-brain-snake
cargo build --release
cargo run --release
Watch a brain grow from random noise into a skilled hunter. It takes time. It's not instant magic — it's evolution. But when you see that snake finally nail a perfect spiral after 100 generations, you'll feel it. Something real just emerged.
