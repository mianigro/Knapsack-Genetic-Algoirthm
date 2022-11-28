import pandas as pd
import streamlit as st
from class_algorithm import GeneticAlgorithm


# Setup page
st.title("Knapsack Problem")
st.subheader("Using a genetic algorithm to optimise a solution.")

# Sidebar with use input
st.sidebar.header("Knapsack Properties")
n_sols = st.sidebar.text_input("Number of solutions")
max_weight = st.sidebar.text_input("Maximum weight")
n_items = st.sidebar.text_input("Number of items")
generations_run = int(st.sidebar.text_input("Number of generations", value=300))
run_alg = st.sidebar.button("Run Algorithm")

# Run algorithm
if run_alg:
    # Progress bar
    st.text("Optimising...")
    my_bar = st.progress(0)

    # Setup genetic algorithm
    alg_gen = GeneticAlgorithm(int(n_sols), int(max_weight), int(n_items))
    new_seed = alg_gen.gen_seed()
    best_fitness_overall = 0

    # Chart the results
    st.header("Fitness Results")
    fitness_list = []
    plot1 = st.line_chart(fitness_list, use_container_width=True)

    # Loop through amount of solutions
    for i in range(generations_run):
        step = 100/generations_run
        # 1. Start with updated steed
        seed = new_seed

        # 2. Put through fitness function to determine fitness
        fitness, solution_array = alg_gen.test_solutions(seed)

        # 3. Select the best solutions
        best_fitness, best_sols = alg_gen.order_fitness(fitness, solution_array)

        # 4. Crossover solutions
        crossed_sols = alg_gen.crossover_function(best_sols)

        # 5. Mutate solutions
        new_seed = alg_gen.mutation(crossed_sols)

        # Add the new results to the graph and update progress
        plot1.add_rows([best_fitness[0]])
        my_bar.progress(int(i * step) + 1)

    sol_dict = {}
    for k, v in enumerate(solution_array[0]):
        sol_dict[f"Item: {k}"] = bool(v)

    st.text("Best solution...")
    sol_df = pd.DataFrame([sol_dict])
    st.dataframe(sol_df)

