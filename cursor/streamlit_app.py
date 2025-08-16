import streamlit as st
import numpy as np
from snake_game import SnakeGame
import time

# Initialize session state
if "game" not in st.session_state:
    st.session_state.game = SnakeGame()

game = st.session_state.game

st.title("🐍 Snake Game with Streamlit")

# Display score and game status
col1, col2 = st.columns(2)
with col1:
    st.metric("Score", game.score)
with col2:
    if game.game_over:
        st.error("Game Over!")
    else:
        st.success("Playing")

# Controls
st.subheader("Controls")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅️ Left"):
        game.change_direction((-1, 0))
with col2:
    col_up, col_down = st.columns(2)
    with col_up:
        if st.button("⬆️ Up"):
            game.change_direction((0, -1))
    with col_down:
        if st.button("⬇️ Down"):
            game.change_direction((0, 1))
with col3:
    if st.button("➡️ Right"):
        game.change_direction((1, 0))

if st.button("🔄 Restart Game"):
    game.reset()
    st.rerun()

# Step the game
game.step()
state = game.get_state()

# Create the game board
board = np.zeros((state["height"], state["width"], 3), dtype=np.uint8)

# Draw snake (green)
for x, y in state["snake"]:
    board[y, x] = [0, 255, 0]  # Green for snake

# Draw food (red)
fx, fy = state["food"]
board[fy, fx] = [255, 0, 0]  # Red for food

# Display the game board
st.subheader("Game Board")
st.image(board, width=400, caption="Green = Snake, Red = Food")

# Game instructions
with st.expander("How to Play"):
    st.write("""
    - Use the arrow buttons to control the snake
    - Eat the red food to grow and increase your score
    - Avoid hitting the walls or yourself
    - Press Restart to start a new game
    """)

# Auto-refresh every 0.5 seconds for continuous gameplay
time.sleep(0.5)
st.rerun() 