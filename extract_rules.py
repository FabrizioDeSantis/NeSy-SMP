from clause import Learner, Options
from c_clause import Loader

path_train = f"rules/pkg.txt"
path_rules_output = f"rules/process-rule-file.txt"

options = Options()

## ANYBURL
options.set("learner.mode", "anyburl")


options.set("learner.anyburl.time", 30)
options.set("learner.anyburl.raw.MAX_LENGTH_CYCLIC", 5)
options.set("learner.anyburl.raw.WORKER_THREADS", 2)

# ## AMIE

# options.set("learner.mode", "amie")

# ## example parameters - choose any supported AMIE options under key "raw"
# # rule length (head+body atom)
# options.set("learner.amie.raw.maxad", 4)
# options.set("learner.amie.raw.mins", 2)
# # special syntax for enforcing -const to be used as flag
# options.set("learner.amie.raw.const", "*flag*")
# # rule length (head+body atom) for rules with constants
# options.set("learner.amie.raw.maxadc", 3)

learner = Learner(options=options.get("learner"))
learner.learn_rules(path_data=path_train, path_output=path_rules_output)