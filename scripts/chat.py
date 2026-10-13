from acme_supportos.models import estimate_cost, get_model, model_name
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM = SystemMessage("You are Acme Cloud support.")
history = [SYSTEM]

model = get_model()
name = model_name()
total_in = total_out = 0
total_cost = 0.0


def print_totals():
    print(f"Session total: in={total_in} out={total_out} tokens, ~${total_cost:.5f}")


try:
    while True:
        try:
            user_input = input("User> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input == "/reset":
            history = [SYSTEM]
            print("History cleared.")
            continue

        history.append(HumanMessage(user_input))
        print("bot> ", end="", flush=True)
        response_final = None
        try:
            for chunk in model.stream(history):
                print(chunk.text, end="", flush=True)
                response_final = chunk if response_final is None else response_final + chunk
        except KeyboardInterrupt:
            history.pop()  # drop the unanswered user message
            print("\n[interrupted]")
            continue
        print()
        if response_final is None:
            history.pop()
            continue
        history.append(response_final)

        usage = response_final.usage_metadata or {}
        turn_in = usage.get("input_tokens", 0)
        turn_out = usage.get("output_tokens", 0)
        turn_cost = estimate_cost(name, turn_in, turn_out)
        total_in += turn_in
        total_out += turn_out
        total_cost += turn_cost
        print(
            f"[turn: in={turn_in} out={turn_out} ~${turn_cost:.5f}] "
            f"[total: in={total_in} out={total_out} ~${total_cost:.5f}]"
        )
except KeyboardInterrupt:
    print()

print_totals()
