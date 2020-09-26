---
layout: post
title:  "[Short post] Thoughts on parsing"
date:   2020-09-1 08:19:16 +0000
categories: jekyll update
---
## Intro
 
A small post with the desire of compiling thoughts on parsing/grammar and all that. This is mostly for myself and you're gonna have a better time learning the topic from someone else.
 
## Definition
 
Usually by parsing people mean deriving a pointery like tree structure off of linear text. Why is it useful? Because apparently sequences/texts/formulae and anything really with syntactic structure is easy for our brains. So much that some say the ability to reason in sentences using recursive derivation laws is the main differential property of human brains compared to other beings. That might be the truth as well as false; doesn't matter, what matters is that syntactic reasoning is important and somewhat fundamental. But true power though is not in the theoretical plane, rather, it's purely practical, this framework allows one to define highly densely packed descriptions of whichever logic they might want it to.
 
## Examples
 
We're gonna go with context free stuff here; meaning that a target interpreter is going to derive a tree based only on a handful of characters to look up for(further as well as backwards).
 
```
def a : i32 = 0;
i32 a = 0;
var a = 0;
(def a (make-i32 0))
let a : i32 = 0 as i32;
```
Those all are pretty clear. We declare an integer and initialize it with zero.
 
It's too boring to go with one character at a time so I'll use 'tokenization' even though it's not necessary in code. Let's look at the tokens of the first line: [`def`, `a`, `:`, `i32`, `=`, `0`, `;`]. The theoretical parsing is usually defined backwards via the construction rules. These rules tell what structures could be generated in a given language.
 
```js
PROGRAM    =   null
             | STATEMENT PROGRAM
 
STATEMENT  =   'def' SYMBOL ':' SYMBOL ';'
             | 'def' SYMBOL ':' SYMBOL '=' EXPRESSION ';'
 
EXPRESSION = INTEGER_VALUE | EXPRESSION BINOP EXPRESSION | SYMBOL
 
SYMBOL             = '[a-zA-Z_]+[a-zA-Z_0-9]*'
INTEGER_VALUE      = `[0-9]+`
BINOP              = '+' | '-' | '*' | '/'
```
This description says that our program is either empty or it consists of a statement followed by a program. The statement is a declaration of a variable that is either initialized by an expression or not. It's quite simple but already embeds some variations like `def a : i32 = 0; def b : MyType = a + 0;`. This rule doesn't understand the difference between names and type names; this is taken care of at semantic analysis stage.
 
Recursive nature of the production rules is the core of its power. It would be an interesting theoretical topic by itself but it's also super useful in a lot of applications. More specifically the production rules could be used to write a parser, even though the job of the parser is to reverse the production rules the structure of both is highly correlated.
 
This is how a parser for the rule might look like in c++
```c++
Expr *parse_expression(char const *&cursor) {
    Expr *lhs   = NULL; // Parse left hand side of the expression first
    auto integer_value = get_next_integer_value(cursor);
    if (integer_value.isNotEmpty()) {
        lhs        = new Expr;
        lhs->type  = eIntegerValue;
        lhs->v_i32 = integer_value.getI32();
    } else {
        auto symbol = get_next_symbol(cursor);
        if (symbol.isNotEmpty()) {
            lhs          = new Expr;
            expr->type   = eSymbol;
            expr->symbol = symbol;
        }
    }
    expect(lhs != NULL);
    auto operator = get_next_binoperator(cursor);
    if (operator.isNotEmpty()) { // Optionally parse a binary expression
        Expr *rhs = parse_expression(cursor); // recursion
        expect(rhs != NULL);
        Expr *op = new Expr;
        op->type = eBinOP;
        op->op   = operator;
        op->lhs  = lhs;
        op->rhs  = rhs;
        // Fix the precedence
        if (op->lhs && op->rhs && rhs->type == eBinOp) {
            // Simpe tree rotation
            // All the subtrees are already balanced
            // So we only need to take care of the current children
            // a0 * a1 + a2
            // we parse:         we need:
            //    *                  +  
            //  /  \                / \
            // a0   +     ---->    *   a2
            //     / \            / \
            //    a1  a2         a0  a1
            //
            if (getPrecedence(rhs->op) < getPrecedence(op->op)) {
                Expr *a0 = op->lhs;
                Expr *a1 = rhs->lhs;
                Expr *a2 = rhs->rhs;
                expect(a0 && a1 && a0);
                SWAP(op->op, rhs->op);
                rhs->lhs = a0;
                rhs->rhs = a1;
                op->lhs  = rhs;
                op->rhs  = a2;
            }
        }
        return op;
    }
    return lhs;
}
Statement *parse_statement(char const *&cursor) {
    auto token = get_next_token(cursor);
    if (token.isEmpty()) // An empty program
        return NULL;
    if (token == "def") {
        // rule: 'def' SYMBOL ':' SYMBOL (';' | '=' EXPRESSION ';')
        auto name = get_next_symbol(cursor);   // get name
        expect(name.isNotEmpty());
        expect(get_next_token(cursor) == ':');
        auto type = get_next_symbol(cursor);   // get type name
        expect(type.isNotEmpty());
        auto continuation = get_next_token(cursor);
        expect(continuation == "=" || continuation == ";");
        Expr *init = NULL;
        if (continuation == "=") {             // does the declaration have an initialization?
            init = parse_expression(cursor);
            expect(init != NULL);
        }
        Statement *decl = new Statement;
        decl->type     = eDeclaration;
        decl->name     = name;
        decl->typename = type;
        decl->init     = init;
        decl->next     = parse_statement(cursor); // recursion
        return decl;
    } else if (...) {} // Some other cases
    return NULL;
}
```
`get_next_token`, `get_next_symbol` etc. are not that interesting and are left as an exercise for the reader. This example aims to illustrate that the parser structure mirrors the structure of the production rules, that's one of  reasons it's worth considering them.
 
## The consistency
 
The good ol tools(bison, flex, lex etc) allow for contradiction detection i.e. finding an example of ambiguity in your production rules. Mindlessly adding stuff to production rules usually doesn't work. That's why it's also useful to reason about the grammar first before writing a parser. For example if your grammar allows attributes you might declare types like that:
 
```js
TYPE        = ATTRIBUTE TYPENAME | TYPENAME
FUNCTION    = ATTRIBUTE TYPE NAME '(' PARAMETERS ')' '{' STATEMENT_LIST '}'
PARAMETERS  = PARAMETER | PARAMETER ',' PARAMETERS
PARAMETER   = TYPE NAME
ATTRIBUTE   = '[' '[' STRING ']' ']'
```
 
It's inconsistent in the cases like `[["attribute"]] void function (int a, int b) {}` because it's ambiguous whether the attribute is of function or of its return type.


## Applications
Nowadays domain specific languages are everywhere. If you work in graphics you're using at least 3 languages.
 
1. C++
2. HLSL
3. Build system 
 
And the list only expands as the system you're working on becomes more and more complex. The languages are there not to increase the complexity though but to lower it.
They are there to encode semantics and get rid of repetition. Naturally, their goal is to increase the entropy of the code. Somehow high entropy encoding is easier to manage simply because the amount of errors is somewhat proportional to the volume of code and anything that decreases the volume decreases the possibility of an error. Low code volume is also easier to change. That's mostly a speculation, of course some stream of symbols like 'egasoggoGEge0g20830t-=' might encode a lot of useful logic but it's non readable and a more readable version is preferred even though it is of lower entropy. 
