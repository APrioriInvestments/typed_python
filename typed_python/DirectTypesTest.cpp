#include "DirectTypes.hpp"
#include "DirectTypesTest.hpp"

#define my_assert(cond) if (!(cond)) { std::cerr << "  failure: " << #cond << std::endl << "  " << __FILE__ << " line " << std::dec << __LINE__ << std::endl; return 1; }
#define test_fn_header() std::cerr << " Start " << __FUNCTION__ << "()" << std::endl;

void errlog(std::string a) { std::cerr << a << std::endl; }

int test_string() {
    test_fn_header()

    String s1("abc");
    StringType::layout* l = s1.getLayout();
    my_assert(l->pointcount == 3)
    my_assert(l->bytes_per_codepoint == 1)
    std::string s2a("abc");

    String s2(s2a);
    my_assert(s1 == s2)
    {
        String s3(s2);
        my_assert(s3 == s2)
        String s4 = s2;
        my_assert(s4 == s2)
        my_assert(s1.getLayout()->refcount == 1)
        my_assert(s2.getLayout()->refcount == 3)
        my_assert(s3.getLayout()->refcount == 3)
        my_assert(s4.getLayout()->refcount == 3)
    }
    my_assert(s1.getLayout()->refcount == 1)
    my_assert(s2.getLayout()->refcount == 1)

    s2 = s1.upper();
    my_assert(s2 == String("ABC"))
    my_assert(s2.getLayout()->refcount == 1)
    s2 = s2.lower();
    my_assert(s2 == String("abc"))
    my_assert(s2.getLayout()->refcount == 1)

    String s3("abcxy");
    my_assert(s3.getLayout()->pointcount == 5)
    my_assert(s3.find(String("cx")) == 2)
    my_assert(s3.substr(1,4) == String("bcx"))

    String a1("DEF");
    String a2("ghij");
    String a3("K");
    String a4("mno");
    String s("blank");
    s = a1 + a2;
    my_assert(s == String("DEFghij"))
    my_assert(s.getLayout()->refcount == 1)
    s = a1 + a2 + a3;
    my_assert(s == String("DEFghijK"))
    my_assert(s.getLayout()->refcount == 1)
    s = a1 + a2 + a3 + a4;
    my_assert(s == String("DEFghijKmno"))
    my_assert(s.getLayout()->refcount == 1)

    String s4("test");
    s4 = s4;
    my_assert(s4.getLayout()->refcount == 1)
    s4 = String("12345");
    my_assert(s4.getLayout()->refcount == 1)
    s4 = String("aaaaaaaaaaaaaaaa");
    my_assert(s4.getLayout()->refcount == 1)
    s1 = String("aBc");
    s2 = String("123X56");
    s2 = String("123");
    s2 = String("xyz");
    s2 = String("ASD");

    my_assert(String("aBc").isalpha())
    my_assert(!String("a3Bc").isalpha())
    my_assert(String("a3Bc").isalnum())
    my_assert(!String("aB%c").isalnum())
    my_assert(String("43234").isdecimal())
    my_assert(!String("43r234").isdecimal())
    my_assert(String("43234").isdigit())
    my_assert(!String("43r234").isdigit())
    my_assert(String("abc 2").islower())
    my_assert(!String("4rL34").islower())
    my_assert(String("43234").isnumeric())
    my_assert(!String("43r234").isnumeric())
    my_assert(String("efg43234").isprintable())
    my_assert(!String("\x01").isprintable())
    my_assert(String("\n \r").isspace())
    my_assert(!String("\n A\r").isspace())
    my_assert(String("One Two").istitle())
    my_assert(!String("OnE Two").istitle())
    my_assert(String("OET").isupper())
    my_assert(!String("OnE Two").isupper())

    my_assert(String("a") > String("A"))
    my_assert(String("abc") < String("abcd"))

    String s5("one two three four");
    ListOf<String> parts = s5.split();
    my_assert(parts.count() == 4)
    my_assert(parts[0] == String("one"))
    my_assert(parts[1] == String("two"))
    my_assert(parts[2] == String("three"))
    my_assert(parts[3] == String("four"))
    parts = s5.split(String("t"));
    my_assert(parts.count() == 3)
    my_assert(parts[0] == String("one "))
    my_assert(parts[1] == String("wo "))
    my_assert(parts[2] == String("hree four"))
    parts = s5.split(2);
    my_assert(parts.count() == 3)
    parts = s5.split(String("t"), 1);
    my_assert(parts.count() == 2)
    return 0;
}

int test_list_of() {
    test_fn_header()

    ListOf<int64_t> list1;
    my_assert(list1.count() == 0)
    my_assert(list1.getLayout()->refcount == 1)

    ListOf<int64_t> list2({9, 8, 7});
    my_assert(list2.count() == 3)
    list2.append(6);
    my_assert(list2.count() == 4)
    my_assert(list2.getLayout()->refcount == 1)

    {
        ListOf<int64_t> list3 = list2;
        my_assert(list2.getLayout()->refcount == 2)
        ListOf<int64_t> list4(list3);
        my_assert(list2.getLayout()->refcount == 3)
    }
    my_assert(list2.getLayout()->refcount == 1)
    my_assert(list2[2] == 7)
    list2[2] = 42;
    my_assert(list2[2] == 42)

    ListOf<String> list4({String("ab"), String("cd")});
    my_assert(list4.count() == 2)
    my_assert(list4[0] == String("ab"))
    my_assert(list4[1] == String("cd"))
    list4[1] = String("replace");
    my_assert(list4[1] == String("replace"))
    return 0;
}

int test_tuple_of() {
    test_fn_header()

    TupleOf<int64_t> t1;
    my_assert(t1.count() == 0)

    TupleOf<int64_t> t2({10, 12, 14, 16});
    my_assert(t2.getLayout()->refcount == 1)
    my_assert(t2.count() == 4)
    my_assert(t2.getLayout()->refcount == 1)

    {
        TupleOf<int64_t> t3 = t2;
        my_assert(t2.getLayout()->refcount == 2)
        TupleOf<int64_t> t4(t3);
        my_assert(t2.getLayout()->refcount == 3)
    }
    my_assert(t2.getLayout()->refcount == 1)
    my_assert(t2[2] == 14)
    return 0;
}

int test_dict() {
    test_fn_header()

    Dict<bool, int64_t> d1;
    d1.insertKeyValue(false, 13);
    d1.insertKeyValue(true, 17);
    const int64_t *pv = d1.lookupValueByKey(false);
    my_assert(pv);
    my_assert(*pv == 13);
    pv = d1.lookupValueByKey(true);
    my_assert(pv);
    my_assert(*pv == 17);
    my_assert(d1.getLayout()->refcount == 1)
    bool deleted = d1.deleteKey(true);
    my_assert(deleted)
    pv = d1.lookupValueByKey(true);
    my_assert(pv == 0)

    {
        Dict<bool, int64_t>d2 = d1;
        Dict<bool, int64_t>d3 = d2;
        my_assert(d1.getLayout()->refcount == 3)
    }
    my_assert(d1.getLayout()->refcount == 1)

    Dict<String, String> d4;
    d4.insertKeyValue(String("first"), String("1st"));
    d4.insertKeyValue(String("second"), String("2nd"));
    d4.insertKeyValue(String("third"), String("3rd"));
    const String *ps = d4.lookupValueByKey(String("second"));
    my_assert(ps);
    my_assert(*ps == String("2nd"));
    return 0;
}

int test_one_of() {
    test_fn_header()

    OneOf<bool, String> o1(true);
    my_assert(o1.getLayout()->which == 0)

    OneOf<bool, String> o2(String("true"));
    my_assert(o2.getLayout()->which == 1)

    OneOf<bool, String, TupleOf<int>> o3;
    bool b;
    String s;
    TupleOf<int> t;
    o3 = TupleOf<int>({9,8,7});
    my_assert(o3.getLayout()->which == 2)
    my_assert(!o3.getValue(b));
    my_assert(!o3.getValue(s));
    my_assert(o3.getValue(t));
    my_assert(t.count() == 3);
    my_assert(t[0] == 9 && t[1] == 8 && t[2] == 7);

    o3 = false;
    my_assert(o3.getLayout()->which == 0)
    my_assert(!o3.getValue(s));
    my_assert(!o3.getValue(t));
    my_assert(o3.getValue(b))
    my_assert(b == false)

    o3 = String("yes");
    my_assert(o3.getLayout()->which == 1)
    my_assert(!o3.getValue(b))
    my_assert(!o3.getValue(t));
    my_assert(o3.getValue(s));
    my_assert(s == String("yes"))

    // fails:
//    OneOf<OneOf<bool, String>, OneOf<int32_t, String>> o4;
//    o4 = String("a");
//    o4 = (int32_t)9;
//    o4 = true;
    return 0;
}

int test_list_of() {
    test_fn_header()
    ListOf<int64_t> list1;
    my_assert(list1.count() == 0)
    my_assert(list1.getLayout()->refcount == 1)
    ListOf<int64_t> list2({9, 8, 7});
    my_assert(list2.count() == 3)
    list2.append(6);
    my_assert(list2.count() == 4)
    my_assert(list2.getLayout()->refcount == 1)
    {
        ListOf<int64_t> list3 = list2;
        my_assert(list2.getLayout()->refcount == 2)
        ListOf<int64_t> list4(list3);
        my_assert(list2.getLayout()->refcount == 3)
    }
    my_assert(list2.getLayout()->refcount == 1)
    my_assert(list2[2] == 7)
    list2[2] = 42;
    my_assert(list2[2] == 42)
    return 0;
}

int test_tuple_of() {
    test_fn_header()
    TupleOf<int64_t> t1;
    my_assert(t1.count() == 0)
    TupleOf<int64_t> t2({10, 12, 14, 16});
    my_assert(t2.getLayout()->refcount == 1)
    my_assert(t2.count() == 4)
    my_assert(t2.getLayout()->refcount == 1)
    {
        TupleOf<int64_t> t3 = t2;
        my_assert(t2.getLayout()->refcount == 2)
        TupleOf<int64_t> t4(t3);
        my_assert(t2.getLayout()->refcount == 3)
    }
    my_assert(t2.getLayout()->refcount == 1)
    my_assert(t2[2] == 14)
    return 0;
}

int direct_cpp_tests() {
    int ret = 0;
    std::cerr << "Start " << __FUNCTION__ << "()" << std::endl;
    ret += test_string();
    ret += test_list_of();
    ret += test_tuple_of();
    ret += test_dict();
    ret += test_one_of();

    std::cerr << ret << " test" << (ret == 1 ? "" : "s") << " failed" << std::endl;

    return ret > 0;
}
