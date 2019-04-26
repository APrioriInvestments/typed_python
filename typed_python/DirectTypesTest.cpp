#include "DirectTypes.hpp"
#include "DirectTypesTest.hpp"

#define my_assert(cond) if (!(cond)) { std::cerr << " failure at " << __FILE__ << " line " << std::dec << __LINE__ << std::endl; return 1; }

int test_string() {
    std::cerr << "start " << __FUNCTION__ << std::endl;
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
    return 0;
}

int direct_cpp_tests() {
    int ret = 0;
    std::cerr << "start " << __FUNCTION__ << std::endl;
    ret += test_string();

    std::cerr << ret << " test" << (ret == 1 ? "" : "s") << " failed" << std::endl;

    return ret > 0;
}
