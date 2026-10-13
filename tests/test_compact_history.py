import ast
import unittest
from pathlib import Path

# Execute only the history-compaction method, independent of third-party SDKs.
source=ast.parse((Path(__file__).resolve().parents[1]/'agent/werewolf_agent.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='WerewolfAgent')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compact_history')
namespace={}
exec(compile(ast.Module(body=[method],type_ignores=[]),'compact_history','exec'),namespace)

class CompactTests(unittest.TestCase):
    def test_bounds_private_preservation_and_order(self):
        history=[{'day':1,'sender':i%7,'recipients':[0] if i==0 else [],'text':str(i)+'x'*300} for i in range(30)]
        agent=type('Agent',(),{'history':history})()
        result=namespace['_compact_history'](agent)
        self.assertLessEqual(sum(len(m['text']) for m in result),1800)
        self.assertTrue(all(len(m['text'])<=240 for m in result))
        self.assertEqual(result[-1]['text'],history[-1]['text'][:240])
        # Four short private messages survive alongside short recent public ones.
        agent.history=[{**m,'text':str(i)} for i,m in enumerate(history)]
        result=namespace['_compact_history'](agent)
        self.assertEqual(result[0]['recipients'],[0])
        self.assertEqual([int(m['text']) for m in result],sorted(int(m['text']) for m in result))
        self.assertEqual(len(agent.history),30)

if __name__=='__main__':unittest.main()
