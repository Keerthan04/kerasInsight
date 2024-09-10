'use client'
import React from "react";
import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  VStack,
  HStack,
  Image,
  useColorModeValue,
} from "@chakra-ui/react";
import { ChevronRight } from "lucide-react";
import Link from "next/link";

export default function LandingPage() {
  const bgColor = useColorModeValue("white", "gray.800");
  const textColor = useColorModeValue("gray.800", "white");
  const redColor = useColorModeValue("red.500", "red.300");

  return (
    <Box bg={bgColor} minHeight="100vh">
      <Container maxW="container.xl" py={20}>
        <VStack spacing={10} align="stretch">
          <HStack justify="space-between" align="center">
            <Image src="/api/placeholder/150/50" alt="Logo" />
            <Button variant="outline" colorScheme="red">
              Login
            </Button>
          </HStack>

          <VStack spacing={6} align="center" textAlign="center">
            <Heading as="h1" size="3xl" color={redColor}>
              KerasInsight-API Documentation Assistant
            </Heading>
            <Text fontSize="xl" color={textColor} maxW="2xl">
              Harness the power of AI to navigate and understand Keras API
              documentation with ease. Get instant answers to your queries and
              streamline your development process.
            </Text>
            <Link href="/dashboard">
              <Button
                rightIcon={<ChevronRight />}
                colorScheme="red"
                size="lg"
                fontSize="md"
                fontWeight="bold"
                px={8}
              >
                Get Started
              </Button>
            </Link>
          </VStack>

          <Box
            borderRadius="lg"
            overflow="hidden"
            boxShadow="2xl"
            bg={useColorModeValue("gray.50", "gray.700")}
          >
            {/* <Image src="/api/placeholder/1200/600" alt="Application Preview" /> */}
          </Box>

          <VStack spacing={8} align="center" pt={10}>
            <Heading as="h2" size="xl" color={textColor}>
              How It Works
            </Heading>
            <HStack spacing={10} align="flex-start">
              {[
                "Input API Endpoints",
                "Create Knowledge Base",
                "Ask Questions",
              ].map((step, index) => (
                <VStack key={index} align="center" maxW="xs">
                  <Box
                    w={16}
                    h={16}
                    borderRadius="full"
                    bg={redColor}
                    color="white"
                    fontSize="2xl"
                    fontWeight="bold"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                  >
                    {index + 1}
                  </Box>
                  <Text fontSize="lg" fontWeight="semibold" color={textColor}>
                    {step}
                  </Text>
                </VStack>
              ))}
            </HStack>
          </VStack>
        </VStack>
      </Container>
    </Box>
  );
}
