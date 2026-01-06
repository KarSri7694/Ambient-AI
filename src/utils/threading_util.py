import threading
import functools
# import memory.db as db  <-- If you want to log completion to DB

def run_async(func):
    """
    Decorator that runs the decorated function in a background thread.
    Returns a generic 'Started' message immediately to the caller.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Define the worker that runs in the void
        def worker_logic():
            try:
                # Run the actual function
                result = func(*args, **kwargs)
                print(f"✅ Background Task '{func.__name__}' finished.")                
            except Exception as e:
                print(f"❌ Background Task '{func.__name__}' failed: {e}")

        # 2. Spawn the thread
        t = threading.Thread(target=worker_logic)
        # t.daemon = True
        t.start()
        
        # 3. Return immediately to the LLM
        return f"Started background task: '{func.__name__}'. I will notify you when it is done."
    
    return wrapper